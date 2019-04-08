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
from .calibration import Calibration
from .constants import CEOUT, BAD_PIXEL
from .mathfunc import get_mythen_data, get_data_without_noise
from .specreader import read_scan, get_filename


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

    def calc_spectra(self, scan_file, scan_id):
        self._scan_file = scan_file
        self._scan_id = scan_id
        self.log.info('Reading scan {} from {}'.format(scan_id, scan_file))
        scan_data, scan_snapshot = read_scan(scan_file, scan_id)
        m_data = get_mythen_data(scan_data)
        m_wn = get_data_without_noise(m_data, self._calib.roi_low,
                                      self._calib.roi_high)

        energies = np.array(scan_data[CEOUT])
        p0_e = self._calib.energy2pixel(energies).astype(int)
        p0_delta = p0_e - self._calib.p0
        min_scale = 0 + p0_delta.min()
        max_scale = BAD_PIXEL + p0_delta.max()
        pixel_scale = np.array(range(min_scale, max_scale))
        scan_sum = np.zeros(pixel_scale.shape[0])
        for p0_d, i in zip(p0_delta, m_wn):
            min_pixel = 0 + p0_d + np.abs(min_scale)
            max_pixel = min_pixel + BAD_PIXEL
            scan_sum[min_pixel:max_pixel] = scan_sum[min_pixel:max_pixel] + i

        self.energy_scale = self._calib.pixel2energy(pixel_scale)
        self.spectra = scan_sum

    def save_to_file(self, output_file):
        plot_filename = get_filename(output_file, suffix='plot')
        plot_data = np.array([self.energy_scale, self.spectra])
        header = 'S 1 Spectra plot\n' \
                 'C Calibration file: {}\n'.format(self._calib_file) +\
                 'C Scan file: {}\n'.format(self._scan_file) + \
                 'C Scan ID: {}\n'.format(self._scan_id) + \
                 'N 2\n' \
                 'L  energy  spectra'
        np.savetxt(plot_filename, plot_data.T, header=header, comments='#')
        self.log.info('Saved spectra plot: {}'.format(plot_filename))


def main(scan_file, scan_id, calib_file, output_file):
    spectra = Spectra(calib_file)
    spectra.calc_spectra(scan_file, scan_id)
    spectra.save_to_file(output_file)
