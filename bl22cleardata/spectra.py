# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# This file is part of BL22 Mythen Post Processing ()
#
# Author(s): Dominique Heinis <dheinis@cells.es>,
#            Roberto J. Homs Puron <rhoms@cells.es>
#
# Copyright 2008-2017 CELLS / ALBA Synchrotron, Bellaterra, Spain
#
# Distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
#
# You should have received a copy of the GNU General Public License
# along with the software. If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import logging
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from multiprocessing import Process
from .calibration import Calibration
from .constants import CEOUT, BAD_PIXEL
from .tool import get_mythen_data, normalize, save_mythen_raw, save_plot
from .specreader import read_scan


class Spectra:
    """
    Class to extract the kBeta spectra
    """

    def __init__(self, calib_file):
        """
        Class to calculate the Spectra
        :param calib_file: json file with the calibration.
        """
        self.log = logging.getLogger('bl22cleardata.Spectra')
        self._calib = Calibration(calib_file)
        self._calib_file = calib_file
        self.spectra = None
        self.energy_scale = None
        self._scan_file = None
        self._scan_id = None
        self.m_data = None

    def calc_spectra(self, scan_file, scan_id, show_plot=False):
        self._scan_file = scan_file
        self._scan_id = scan_id
        self.log.info('Reading scan {} from {}'.format(scan_id, scan_file))
        scan_data, scan_snapshot = read_scan(scan_file, scan_id)
        m_data = get_mythen_data(scan_data)
        self.m_data = m_data

        # Calculate the continue energy scale
        energies = np.array(scan_data[CEOUT])
        p0_e = self._calib.energy2pixel(energies).astype(int)
        p0_delta = p0_e - self._calib.p0
        min_pixel_scale = 0 + p0_delta.min()
        max_pixel_scale = BAD_PIXEL + p0_delta.max()
        min_energy_scale = self._calib.pixel2energy(max_pixel_scale)
        max_energy_scale = self._calib.pixel2energy(min_pixel_scale)
        energy_step = int(abs((max_energy_scale - min_energy_scale)
                              / self._calib.energy_resolution))
        cont_energy_scale = np.linspace(min_energy_scale, max_energy_scale,
                                        energy_step)

        nr_points = m_data.shape[0]
        intensity_matrix = np.zeros([nr_points, energy_step])
        overlap_vector = np.ones(cont_energy_scale.shape[0])

        # Normalize the mythen calibration between 1 and noise level
        m_calib_norm = normalize(self._calib.m_calib,
                                 scale_min=self._calib.noise_percent/100)

        # TODO: check comments!!!
        # Calculate the resolution. All peaks must be on the same position.
        # Use a interpolation of each image and calculate the resolution
        # according to the energy resolution.
        # To calculate the resolution we use the stadistic sum:
        #
        #    Y = sum( Yi/sigma_i^2) / sum(1/sigma_i^2)
        #    Yi: intensity vector (Mythen line)
        #
        # In this case -> sigma_i = sqrt(Yi) and the Y is:
        #       Y = nr/ sum(1/Yi)
        #       nr: number of images
        # Define Ysum = sum(1/Yi)
        #       Y = nr / Ysum
        # The error: S = 1/sqrt(sum(1/sigma_i^2))
        #       S = 1/sqrt(Ysum)
        # This method is not applicable to the calibration.

        if show_plot:
            calib_middle_scan = None
            m_i_raw_middle_scan = None
            middle_scan_idx = int((m_data.shape[0] - 1)/2)

        Ysum = np.zeros(cont_energy_scale.shape[0])
        for idx, m_i_raw in enumerate(m_data):
            p0_d = p0_delta[idx]

            delta_energy = energies[idx] - self._calib.e0
            calib_energy_scale = self._calib.m_energy_scale + delta_energy

            # Interpolation function for the calibration
            f_calib = interp1d(calib_energy_scale, m_calib_norm,
                               bounds_error=False, fill_value=1)

            # Interpolation function for the calibration window: 1 inside
            # the window and 0 outside
            calib_window = np.ones(m_calib_norm.shape[0])
            f_calib_window = interp1d(calib_energy_scale, calib_window,
                                      bounds_error=False, fill_value=0)

            # Calculate the discrete energy scale for the Mythen RAW data
            # interpolation.
            min_pixel = 0 + p0_d
            max_pixel = min_pixel + BAD_PIXEL
            discrete_energy_scale = self._calib.pixel2energy(range(min_pixel,
                                                                   max_pixel))
            min_energy = discrete_energy_scale.min()
            max_energy = discrete_energy_scale.max()
            lower_indexes = np.where(cont_energy_scale < min_energy)[0]
            higher_indexes = np.where(cont_energy_scale > max_energy)[0]

            try:
                lower_index = lower_indexes.max() + 1
            except Exception:
                lower_index = 0

            try:
                higher_index = higher_indexes.min() - 1
            except Exception:
                higher_index = cont_energy_scale.shape[0] - 1

            f_mythen_raw = interp1d(discrete_energy_scale, m_i_raw,
                                    bounds_error=False, fill_value=0)

            overlap_vector[lower_index:higher_index] += 1

            # Calulate the intensity value Yi:
            # 1) Set to Zero value out of the window by multiplying with
            #    f_calib_window
            # 2) Normalize by the mythen_calibration by dividing per f_calib

            m_i_raw_window = f_mythen_raw(cont_energy_scale) * \
                             f_calib_window(cont_energy_scale)
            Yi = m_i_raw_window / f_calib(cont_energy_scale)

            intensity_matrix[idx] = Yi
            if show_plot and idx == middle_scan_idx:
                calib_middle_scan = f_calib(cont_energy_scale)
                m_i_raw_middle_scan = m_i_raw_window
            Yi = Yi.clip(min=0.0001)
            Ysum += 1.0 / Yi

        Y = overlap_vector / Ysum
        spectra = intensity_matrix.sum(axis=0) / overlap_vector
        spectra_window = np.where(spectra > 0)
        spectra = spectra[spectra_window]
        self.spectra = spectra
        self.energy_scale = cont_energy_scale[spectra_window]
        self.m_data = m_data
        self.scan_sum = Y

        def plot_data():
            # TODO: Evaluate if to use thread to not block
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 1].imshow(m_data[:, self._calib.roi_low:
                                       self._calib.roi_high])

            # Plot the intensity and the calibration window for the middle
            # of the scan.
            color ='tab:red'
            axs01 = axs[0, 0]
            axs01.tick_params(axis='y', labelcolor=color)
            axs01.plot(self.energy_scale, m_i_raw_middle_scan[spectra_window],
                       color=color)
            color = 'tab:blue'
            axs2 = axs[0, 0].twinx()
            axs2.tick_params(axis='y', labelcolor=color)
            axs2.plot(self.energy_scale, calib_middle_scan[spectra_window],
                      color=color)

            spectra = intensity_matrix[middle_scan_idx]
            axs[1, 0].plot(self.energy_scale, spectra[spectra_window])

            axs[1, 1].plot(self.energy_scale, self.spectra,
                           color=color)

            # color = 'tab:red'
            # axs1 = axs[1, 1]
            # axs1.tick_params(axis='y', labelcolor=color)
            #
            # color = 'tab:blue'
            # axs2 = axs[1, 1].twinx()
            # axs2.tick_params(axis='y', labelcolor=color)
            # axs2.plot(self.energy_scale, self.scan_sum, color=color)
            plt.show()

        if show_plot:
            p = Process(None, plot_data)
            p.start()

    def save_to_file(self, output_file, extract_raw=False):

        # TODO: Introduced the scan 0 to do not crash pyMCA. Remove when it
        #  does not fail
        header = 'S 0 No Data\n' \
                 'S 1 Spectra plot ScanID: {}\n'.format(self._scan_id) + \
                 'C Calibration file: {}\n'.format(self._calib_file) + \
                 'C Scan file: {}\n'.format(self._scan_file) + \
                 'C Scan ID: {}\n'.format(self._scan_id) + \
                 'N 2\n' \
                 'L  energy  spectra'
        data = np.array([self.energy_scale, self.spectra])
        save_plot(output_file, data, header, log=self.log)

        if extract_raw:
            save_mythen_raw(output_file, self.m_data, log=self.log)


def main(scan_file, scan_id, calib_file, output_file, show_plot=False,
         extract_raw=False):

    spectra = Spectra(calib_file)
    spectra.calc_spectra(scan_file, scan_id, show_plot=show_plot)
    spectra.save_to_file(output_file, extract_raw=extract_raw)
