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
import json
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from multiprocessing import Process
from .calibration import Calibration
from .constants import CEOUT, BAD_PIXEL, ENERGY
from .tool import get_mythen_data, normalize, save_plot, save_mythen_raw, \
    get_filename
from .specreader import read_scan


class PFY:
    """
    Class to extract the PFY
    """

    def __init__(self, calib_file):
        """
        Class to calculate the Spectra
        :param calib_file: json file with the calibration.
        """
        self.log = logging.getLogger('bl22cleardata.PFY')
        self._calib = Calibration(calib_file)
        self._calib_file = calib_file
        self._scan_file = None
        self._start_scan_id = None
        self._nr_scans = None
        self._ceout = None
        self.pfy = None
        self.energy_scale = None
        self.m_data = None
        self.intensity_matrix = None
        self.mythen_energy_scale = None
        self.energy_roi_low = None
        self.energy_roi_high = None

    def _read_scans(self):
        m_data = None
        energies = None
        for i in range(self._nr_scans):
            scan_id = self._start_scan_id + i
            scan_data, scan_snapshot = read_scan(self._scan_file,
                                                 scan_id)
            self.log.info('Reading scan {} from {}'.format(scan_id,
                                                           self._scan_file))

            m_sub_data = get_mythen_data(scan_data)
            energy_sub_data = scan_data[ENERGY]
            if i == 0:
                m_data = m_sub_data
                energies = energy_sub_data
                self._ceout = scan_snapshot[CEOUT]
            else:
               m_data = np.append(m_data, m_sub_data, axis=0)
               energies = np.append(energies, energy_sub_data)

        self.m_data = m_data
        self.energy_scale = energies

    def calc_pfy(self, scan_file, start_scan_id, nr_scans, user_roi_high=None,
                 user_roi_low=None, show_plot=False):
        self._scan_file = scan_file
        self._start_scan_id = start_scan_id
        self._nr_scans = nr_scans

        # Read the scans normalize each one by I0 and concatenate them.
        self._read_scans()

        # Calculate the Mythen energy scales: discrete and continuous
        p0_e = self._calib.energy2pixel(self._ceout).astype(int)[0]
        p0_delta = p0_e - self._calib.p0
        min_pixel_scale = 0 + p0_delta
        max_pixel_scale = BAD_PIXEL + p0_delta
        min_energy_scale = self._calib.pixel2energy(max_pixel_scale)
        max_energy_scale = self._calib.pixel2energy(min_pixel_scale)
        energy_step = int(abs((max_energy_scale - min_energy_scale)
                              / self._calib.energy_resolution))
        cont_energy_scale = \
            np.linspace(min_energy_scale, max_energy_scale, energy_step)
        discrete_energy_scale = \
            self._calib.pixel2energy(range(min_pixel_scale, max_pixel_scale))

        # Calculate the energy roi, lower pixel higher energies
        pixel_high = self._calib.roi_high + p0_delta
        pixel_low = self._calib.roi_low + p0_delta
        energy_roi_low = self._calib.pixel2energy(pixel_high)
        energy_roi_high = self._calib.pixel2energy(pixel_low)

        if user_roi_high is not None and user_roi_high < energy_roi_high:
            energy_roi_high = user_roi_high
        if user_roi_low is not None and user_roi_low > energy_roi_low:
            energy_roi_low = user_roi_low

        idx = np.where(cont_energy_scale > energy_roi_low)[0]
        idx_energy_roi_low = idx.min()
        idx = np.where(cont_energy_scale < energy_roi_high)[0]
        idx_energy_roi_high = idx.max()

        # Normalize the mythen calibration between 1 and noise level
        m_calib_norm = normalize(self._calib.m_calib,
                                 scale_min=self._calib.noise_percent / 100)
        delta_energy = self._ceout - self._calib.e0
        calib_energy_scale = self._calib.m_energy_scale + delta_energy
        f_calib = interp1d(calib_energy_scale, m_calib_norm,
                           bounds_error=False,
                           fill_value=1)

        # Interpolation function for the calibration window: 1 inside
        # the window and 0 outside
        calib_window = np.ones(m_calib_norm.shape[0])
        f_calib_window = interp1d(calib_energy_scale, calib_window,
                                  bounds_error=False, fill_value=0)

        nr_points = self.m_data.shape[0]
        intensity_matrix = np.zeros([nr_points, energy_step])
        middle_scan_idx = -1
        if show_plot:
            calib_middle_scan = None
            m_i_raw_middle_scan = None
            middle_scan_idx = int((self._m_data.shape[0] - 1)/2)

        for idx, m_i_raw in enumerate(self.m_data):
            # Calculate the discrete energy scale for the Mythen RAW data
            # interpolation.
            f_mythen_raw = interp1d(discrete_energy_scale, m_i_raw,
                                    bounds_error=False, fill_value=0)
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

        pfy_data = intensity_matrix[:, idx_energy_roi_low:idx_energy_roi_high]
        self.pfy = pfy_data.sum(axis=1)
        self.intensity_matrix = intensity_matrix
        self.mythen_energy_scale = cont_energy_scale
        self.energy_roi_low = energy_roi_low
        self.energy_roi_high = energy_roi_high

        def plot_data():
            # TODO: Evaluate if to use thread to not block
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 1].imshow(self.m_data[:, self._calib.roi_low:
                                            self._calib.roi_high])

            # Plot the intensity and the calibration window for the middle
            # of the scan.
            color ='tab:red'
            axs01 = axs[0, 0]
            axs01.tick_params(axis='y', labelcolor=color)
            window = np.where(m_i_raw_middle_scan > 0)

            axs01.plot(cont_energy_scale[window], m_i_raw_middle_scan[window],
                       color=color)
            color = 'tab:blue'
            axs2 = axs[0, 0].twinx()
            axs2.tick_params(axis='y', labelcolor=color)
            axs2.plot(cont_energy_scale[window], calib_middle_scan[window],
                      color=color)

            spectra = intensity_matrix[middle_scan_idx]
            axs[1, 0].plot(cont_energy_scale[window], spectra[window])

            axs[1, 1].plot(self.energy_scale, self.pfy, color=color)
            plt.show()

        if show_plot:
            p = Process(None, plot_data)
            p.start()

    def save_to_file(self, output_file, extract_raw=False,
                     extract_post_process=False):

        # TODO: Introduced the scan 0 to do not crash pyMCA. Remove when it
        #  does not fail
        start_scan_id = self._start_scan_id
        end_scan_id = self._start_scan_id + self._nr_scans
        header = 'S 0 No Data\n' \
                 'S 1 PFY plot ScanIDs: ' \
                 '{} - {}\n'.format(start_scan_id, end_scan_id) + \
                 'C Calibration file: {}\n'.format(self._calib_file) + \
                 'C Scan file: {}\n'.format(self._scan_file) + \
                 'C Energy ROI Low: {}\n'.format(self.energy_roi_low) + \
                 'C Energy ROI High: {}\n'.format(self.energy_roi_high) + \
                 'N 2\n' \
                 'L  energy  pfy'

        data = np.array([self.energy_scale, self.pfy])
        save_plot(output_file, data, header, log=self.log)

        if extract_raw:
            save_mythen_raw(output_file, self.m_data, log=self.log)

        if extract_post_process:
            data = dict()
            data['output_file'] = output_file
            data['energy_scale'] = self.energy_scale.lolist()
            data['mythen_energy_scale'] = self.mythen_energy_scale.tolist()
            data['data_processed'] = self.intensity_matrix.tolist()
            json_file = get_filename(output_file, suffix='json')
            self.log.info('Saving data processed to: '
                          '{}'.format(json_file))
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=0)
            self.log.info('Data saved.')


def main(scan_file, start_scan_id, nr_scans, calib_file, output_file,
         show_plot=False, extract_raw=False, extract_post_process=False,
         user_roi=[None, None]):
    user_roi_low = user_roi[0]
    user_roi_high = user_roi[1]

    pfy = PFY(calib_file)
    pfy.calc_pfy(scan_file, start_scan_id, nr_scans, user_roi_low=user_roi_low,
                 user_roi_high=user_roi_high, show_plot=show_plot)
    pfy.save_to_file(output_file, extract_raw=extract_raw,
                     extract_post_process=extract_post_process)
