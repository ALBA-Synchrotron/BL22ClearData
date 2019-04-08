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
from .constants import ENERGY, CBRAGG, CAZ, get_crystal, BAD_PIXEL
from .specreader import read_scan, get_filename
from .mathfunc import get_mythen_data, linear_regression, calc_autoroi, \
    dispersion_2d, get_best_fit, gauss_function


class Calibration:
    """
    Class to store the calibration
    """

    def __init__(self, calib_file= None):
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
        self.energy_scale = None
        self.scan_resolution = None
        self.scan_sum = None

        if calib_file is not None:
            self.load_from_file(calib_file)

    def load_from_file(self, calib_file):
        self.log.info('Loading calibration from: {}'.format(calib_file))
        with open(calib_file, 'r') as f:
            calib_loaded = json.load(f)
        try:
            self.scan_file = calib_loaded['scan_file']
            self.scan_id = calib_loaded['scan_id']
            self.e0 = calib_loaded['e0']
            self.p0 = calib_loaded['p0']
            self.a = calib_loaded['a']
            self.b = calib_loaded['b']
            self.k = calib_loaded['k']
            self.energy_a = calib_loaded['energy_a']
            self.energy_b = calib_loaded['energy_b']
            self.roi_low = calib_loaded['roi_low']
            self.roi_high = calib_loaded['roi_high']
            self.cbragg_pos = calib_loaded['cbragg_pos']
            self.crystal_order = calib_loaded['crystal_order']
            self.caz_pos = calib_loaded['caz_pos']
            self.crystal = get_crystal(self.caz_pos)
            self.scan_sum = np.array(calib_loaded['scan_sum'])
            self.scan_resolution = np.array(calib_loaded['scan_resolution'])
            self.energy_scale = np.array(calib_loaded['energy_scale'])

        except Exception as e:
            self.log.error('Error on load calibration from '
                           'file: {}'.format(e))
            raise RuntimeError('Wrong calibration')

    def save_to_file(self, output_file):
        calib_to_save = {}
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
        calib_to_save['crystal_order'] = self.crystal_order
        calib_to_save['caz_pos'] = self.caz_pos
        calib_to_save['scan_sum'] = self.scan_sum.tolist()
        calib_to_save['scan_resolution'] = self.scan_resolution.tolist()
        calib_to_save['energy_scale'] = self.energy_scale.tolist()
        calib_filename = get_filename(output_file, suffix='json')
        if self.log.isEnabledFor(logging.DEBUG):
            self.log.debug('Calibration {}'.format(calib_to_save))
        self.log.info('Saving calibration to: {}'.format(calib_filename))
        with open(calib_filename, 'w') as f:
            json.dump(calib_to_save, f, indent=0)
        self.log.info('Calibration saved.')
        plot_filename = get_filename(output_file, suffix='plot')
        plot_data = np.array([self.energy_scale, self.scan_sum,
                              self.scan_resolution])
        header = 'S 1 Calibration plot\n' \
                 'C Calibration file {}\n'.format(calib_filename) +\
                 'N 2\n' \
                 'L  energy  calibration  resolution'
        np.savetxt(plot_filename, plot_data.T, header=header, comments='#')
        self.log.info('Saved calibration plot: {}'.format(plot_filename))

    def pixel2energy(self, pixels):
        return self.energy_a * pixels + self.energy_b

    def energy2pixel(self, energies):
        return (energies - self.energy_b) / self.energy_a

    def calibrate(self, scan_file, scan_id, auto_roi=True,
                  user_roi=[0, BAD_PIXEL], threshold=0.7,
                  noise_percent=2.5):

        self.auto_roi = auto_roi
        self.scan_id = scan_id
        self.scan_file = scan_file

        data, snapshots = read_scan(self.scan_file, self.scan_id, self.log)

        # Get the crystal
        self.caz_pos = snapshots[CAZ][0]
        self.crystal = get_crystal(self.caz_pos)

        # Calculate the order
        energies = np.array(data[ENERGY])
        cbragg_pos = snapshots[CBRAGG]
        self.crystal_order = self.crystal.find_order(energies.mean(),
                                                     cbragg_pos)

        # Calculate a and b for energy scale
        m_data = get_mythen_data(data)
        a, b, x_mean, y_mean, x_std, y_std = linear_regression(m_data,
                                                               threshold)

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
        m_wn = m_data - noise
        m_wroi = m_wn[:, self.roi_low:self.roi_high]
        d = m_wroi / m_wroi.sum()

        X, Y = np.meshgrid(np.arange(m_wroi.shape[1]),
                           np.arange(m_wroi.shape[0]))
        cost = lambda v: ((dispersion_2d(a, x_mean, y_mean,
                                         x_std, y_std, X, Y) - d) ** 2).sum()
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

        # Calculate p0 and e0
        x = np.arange(m_wroi.shape[1])
        disp_exp = m_wroi.max(axis=0)
        x_max, x_mean, x_std = get_best_fit(x, disp_exp, gauss_function)
        self.p0 = int(x_mean + self.roi_low)
        self.e0 = self.pixel2energy(self.p0)

        # Calculate energy scale vector
        self.scan_sum = m_wn.sum(axis=0)

        # Calculate the resolution. All peaks must be on the same position.
        p0_e = self.energy2pixel(energies).astype(int)
        p0_delta = p0_e - self.p0
        pixel_scale = np.array(range(BAD_PIXEL))
        scan_resolution = np.zeros(pixel_scale.shape[0])
        for p0_d, i in zip(p0_delta, m_wn):
            min_pixel = abs(p0_d)
            max_pixel = BAD_PIXEL - min_pixel
            if p0_d > 0:
                scan_resolution[0:max_pixel] = \
                    scan_resolution[0:max_pixel] + i[min_pixel:BAD_PIXEL]
            else:
                scan_resolution[min_pixel: BAD_PIXEL] = \
                    scan_resolution[min_pixel: BAD_PIXEL] + i[0: max_pixel]

        self.energy_scale = self.pixel2energy(pixel_scale)
        self.scan_resolution = scan_resolution


def main(scan_file, scan_id, output_file, auto_roi=True,
         user_roi=[0, BAD_PIXEL], threshold=0.7, noise_percent=2.5):

        calib = Calibration()
        calib.calibrate(scan_file, scan_id, auto_roi, user_roi, threshold,
                        noise_percent)
        calib.save_to_file(output_file)
