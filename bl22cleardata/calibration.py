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
import json
import logging
import time
import numpy as np
from scipy.optimize import fmin
from scipy.interpolate import interp1d
from multiprocessing import Process
from matplotlib import pyplot as plt
from .constants import ENERGY, BAD_PIXEL
from .specreader import read_scan, get_filename
from .mathfunc import get_mythen_data, linear_regression, calc_autoroi, \
    dispersion_2d, get_best_fit, gauss_function


class Calibration:
    """
    Class to store the calibration
    """

    def __init__(self, calib_file=None):
        self.log = logging.getLogger('bl22cleardata.Calibration')

        self.e0 = None
        self.p0 = None
        self.a = None
        self.b = None
        self.k = None
        self.roi_low = None
        self.roi_high = None
        self.scan_id = None
        self.scan_file = None
        self.cbragg_pos = None
        self.crystal_order = None
        self.crystal = None
        self.caz_pos = None
        self.energy_a = None
        self.energy_b = None
        self.m_energy_scale = None
        self.m_resolution = None
        self.m_calib = None
        self.m_data = None
        self.auto_roi = None
        self.noise_percent = None
        self.energy_resolution = None

        if calib_file is not None:
            self.load_from_file(calib_file)

    def load_from_file(self, calib_file):
        self.log.info('Loading calibration from: {}'.format(calib_file))
        with open(calib_file, 'r') as f:
            calib_loaded = json.load(f)
        try:
            self.scan_file = calib_loaded['scan_file']
            self.scan_id = calib_loaded['scan_id']
            self.e0 = np.float64(calib_loaded['e0'])
            self.p0 = calib_loaded['p0']
            self.a = calib_loaded['a']
            self.b = calib_loaded['b']
            self.k = calib_loaded['k']
            self.energy_a = np.float64(calib_loaded['energy_a'])
            self.energy_b = np.float64(calib_loaded['energy_b'])
            self.roi_low = calib_loaded['roi_low']
            self.roi_high = calib_loaded['roi_high']
            self.cbragg_pos = calib_loaded['cbragg_pos']
            # self.crystal_order = calib_loaded['crystal_order']
            # self.caz_pos = calib_loaded['caz_pos']
            # self.crystal = #get_crystal(self.caz_pos)
            self.m_calib = np.array(calib_loaded['mythen_calibration'])
            # self.m_resolution = np.array(calib_loaded['mythen_resolution'])
            self.m_energy_scale = np.array(calib_loaded['mythen_energy_scale'])
            self.energy_resolution = calib_loaded['energy_resolution']
            self.noise_percent = calib_loaded['noise_percent']
        except Exception as e:
            self.log.error('Error on load calibration from '
                           'file: {}'.format(e))
            raise RuntimeError('Wrong calibration')

    def save_to_file(self, output_file, extract_raw=False):
        calib_to_save = dict()
        calib_to_save['date'] = time.strftime('%d/%m/%Y %H:%M:%S')
        calib_to_save['scan_file'] = self.scan_file
        calib_to_save['scan_id'] = self.scan_id
        calib_to_save['e0'] = self.e0
        calib_to_save['p0'] = self.p0
        calib_to_save['a'] = self.a
        calib_to_save['b'] = self.b
        calib_to_save['k'] = self.k
        calib_to_save['energy_a'] = self.energy_a
        calib_to_save['energy_b'] = self.energy_b
        calib_to_save['auto_roi'] = self.auto_roi
        calib_to_save['roi_low'] = self.roi_low
        calib_to_save['roi_high'] = self.roi_high
        calib_to_save['cbragg_pos'] = self.cbragg_pos
        # calib_to_save['crystal_order'] = self.crystal_order
        # calib_to_save['caz_pos'] = self.caz_pos
        calib_to_save['energy_resolution'] = self.energy_resolution
        calib_to_save['noise_percent'] = self.noise_percent
        calib_to_save['mythen_calibration'] = self.m_calib.tolist()
        # calib_to_save['mythen_resolution'] = self.m_resolution.tolist()
        calib_to_save['mythen_energy_scale'] = self.m_energy_scale.tolist()
        calib_filename = get_filename(output_file, suffix='json')
        if self.log.isEnabledFor(logging.DEBUG):
            self.log.debug('Calibration {}'.format(calib_to_save))
        self.log.info('Saving calibration to: {}'.format(calib_filename))
        with open(calib_filename, 'w') as f:
            json.dump(calib_to_save, f, indent=0)
        self.log.info('Calibration saved.')
        plot_filename = get_filename(output_file, suffix='plot')
        plot_data = np.array([self.m_energy_scale, self.m_calib,
                              self.m_resolution])
        # TODO: Introduced the scan 0 to do not crash pyMCA. Remove when it
        #  does not fail
        header = 'S 0 No Data\n' \
                 'S 1 Calibration plot scanID: {}\n'.format(self.scan_id) + \
                 'C Calibration file {}\n'.format(calib_filename) + \
                 'N 2\n' \
                 'L  energy  calibration  resolution'
        np.savetxt(plot_filename, plot_data.T, header=header, comments='#')
        self.log.info('Saved calibration plot: {}'.format(plot_filename))
        if extract_raw:
            header = 'Mythen raw data.'
            mythen_filename = get_filename(output_file, suffix='mythen_data')
            raw_data = self.m_data
            np.savetxt(mythen_filename, raw_data, header=header)
            self.log.info('Saved Mythen normalized '
                          'data: {}'.format(mythen_filename))

    def pixel2energy(self, pixels):
        return self.energy_a * pixels + self.energy_b

    def energy2pixel(self, energies):
        return (energies - self.energy_b) / self.energy_a

    def calibrate(self, scan_file, scan_id, auto_roi=True,
                  user_roi=(0, BAD_PIXEL), threshold=0.7,
                  noise_percent=2.5, energy_resolution=0.03, show_plot=False):

        self.auto_roi = auto_roi
        self.scan_id = scan_id
        self.scan_file = scan_file
        self.energy_resolution = energy_resolution
        self.noise_percent = noise_percent
        self.log.info('Reading data...')
        data, snapshots = read_scan(self.scan_file, self.scan_id, self.log)

        # Get the crystal
        # self.caz_pos = snapshots[CAZ][0]
        # self.crystal = get_crystal(self.caz_pos)

        # Calculate the order
        energies = np.array(data[ENERGY])
        # cbragg_pos = snapshots[CBRAGG]
        # self.crystal_order = self.crystal.find_order(energies.mean(),
        #                                              cbragg_pos)
        self.log.info('Calculating energy scale...')
        # Calculate a and b for energy scale
        m_data = get_mythen_data(data)
        a, b, x_mean, y_mean, x_std, y_std = linear_regression(m_data,
                                                               threshold)

        self.log.info('Setting ROIs...')
        # Calculate autoROIs
        if self.auto_roi:
            self.roi_low, self.roi_high = calc_autoroi(m_data, noise_percent)
        else:
            self.roi_low, self.roi_high = user_roi
        # The original code use only the noise form 0 to a pixel_limit_
        # noise 600. Refactor to use the data out of the roi
        if self.roi_low == 0:
            low_noise = 0
        else:
            low_noise = m_data[:, 0:self.roi_low].mean()
        if self.roi_high == BAD_PIXEL:
            high_noise = 0
        else:
            high_noise = m_data[:, self.roi_high:BAD_PIXEL].mean()
        noise = (low_noise + high_noise) / 2

        # Calculate 2D dispersion
        self.log.info('Calculating 2D fitting...')
        m_wn = m_data - noise
        m_wroi = m_wn[:, self.roi_low:self.roi_high]
        d = m_wroi / m_wroi.sum()

        x, y = np.meshgrid(np.arange(m_wroi.shape[1]),
                           np.arange(m_wroi.shape[0]))
        cost = lambda v: ((dispersion_2d(a, x_mean, y_mean,
                                         x_std, y_std, x, y) - d) ** 2).sum()
        new_a, new_x_mean, new_y_mean, new_x_std, new_y_std = \
            fmin(cost, [a, x_mean, y_mean, x_std, y_std])

        new_b = y_mean - new_a * new_x_mean

        if np.abs((new_a - a) / a) < 0.1:
            # Use 2D dispersion fitting
            a = new_a
            b = new_b
            x_mean = new_x_mean
            y_mean = new_y_mean

        self.a = a
        self.b = b

        # Calculate the scale factor index vs energy_mono:
        # energy = k*i + energy[0]
        self.k = (energies[-1] - energies[0]) / float(len(energies) - 1)
        self.energy_a = self.k * self.a
        self.energy_b = self.k * self.b + energies[0]

        self.log.info('Calculating p0 and e0...')
        # Calculate p0 and e0
        x = np.arange(m_wroi.shape[1])
        disp_exp = m_wroi.max(axis=0)
        x_max, x_mean, x_std = get_best_fit(x, disp_exp, gauss_function)
        self.p0 = int(x_mean + self.roi_low)
        self.e0 = self.pixel2energy(self.p0)

        # Calculate energy scale vector
        self.m_calib = m_wn.sum(axis=0)

        self.log.info('Calculating resolution...')

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

        min_scale = self.pixel2energy(self.roi_low)
        max_scale = self.pixel2energy(self.roi_high)

        discrete_energy_scale = self.pixel2energy(range(0, BAD_PIXEL))
        energy_step = int(abs((max_scale - min_scale)/energy_resolution))
        continue_energy_scale = np.linspace(min_scale, max_scale, energy_step)
        nr_points = m_data.shape[0]
        resolution_matrix = np.zeros([nr_points, energy_step])
        f = interp1d(discrete_energy_scale, m_wn.sum(axis=0),
                     bounds_error=False, fill_value=0)
        self.m_calib = f(continue_energy_scale)
        for idx, i in enumerate(m_data):
            e0_delta = self.pixel2energy(m_data[idx].argmax()) - self.e0
            f = interp1d(discrete_energy_scale - e0_delta, i,
                         bounds_error=False, fill_value=0)
            resolution_matrix[idx] = f(continue_energy_scale)
        self.m_resolution = resolution_matrix.sum(axis=0) / nr_points
        self.m_energy_scale = continue_energy_scale
        self.m_data = m_data

        def plot_data():
            # TODO: Evaluate if to use thread to not block
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].imshow(m_wn[:, self.roi_low:self.roi_high])
            axs[0, 1].imshow(resolution_matrix)
            for i in resolution_matrix:
                axs[1, 0].plot(self.m_energy_scale, i)
            color = 'tab:red'
            axs1 = axs[1, 1]
            axs1.tick_params(axis='y', labelcolor=color)
            axs1.plot(self.m_energy_scale, self.m_resolution,
                      color=color)

            color = 'tab:blue'
            axs2 = axs[1, 1].twinx()
            axs2.tick_params(axis='y', labelcolor=color)
            axs2.plot(self.m_energy_scale, self.m_calib, color=color)
            plt.show()

        if show_plot:
            p = Process(None, plot_data)
            p.start()


def main(scan_file, scan_id, output_file, auto_roi=True,
         user_roi=(0, BAD_PIXEL), threshold=0.7, noise_percent=2.5,
         energy_resolution=0.03, show_plot=False, extract_raw=False):

    calib = Calibration()
    calib.calibrate(scan_file, scan_id, auto_roi, user_roi, threshold,
                    noise_percent, energy_resolution, show_plot)
    calib.save_to_file(output_file, extract_raw=extract_raw)
