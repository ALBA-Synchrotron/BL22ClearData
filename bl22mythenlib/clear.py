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

import numpy as np
import logging
from .mythen import Mythen
from crystal import CrystalSi


class Clear(object):
    """
    Class to work with clear
    """

    def __init__(self, rowland_rad=0.5, order=1, bragg_offset=0,
                 crystal=CrystalSi([1, 1, 1])):
        log_name = '{0}.Clear'.format(__name__)
        self.log = logging.getLogger(log_name)
        self.mythen = Mythen()
        self.rowland_rad = rowland_rad
        self.crystal = crystal
        self.order = order
        self.bragg_offset = bragg_offset
        self.a = None
        self.b = None
        self.p0 = None
        self.i0 = None
        self.e0 = None
        self.k = None

    def pixels2energies(self, pixels):
        """
        Method to transform from pixel to energy
        :param pixels: [int] or int
        :return: np.array or np.float64
        """
        pixels = np.array(pixels)
        energies = self.a * pixels + self.b
        return energies

    def elastic_line_calibration(self, mythen_data, energies, clear_bragg,
                                 order=None, threshold=0.7,
                                 pixel_limit_noise=600):
        self.log.debug('Entering on elastic_line_calibration.')
        self.log.debug('mythen_data: {}'.format(mythen_data))
        self.log.debug('energies: {}'.format(energies))
        self.log.debug('clear_bragg: {}'.format(clear_bragg))
        self.log.debug('order: {}'.format(order))
        self.log.debug('threshold: {}'.format(threshold))

        # Calculate parameters for calibration
        # Find the order if it is not passed
        # TODO: this information should pass to the system
        if order is None:
            bragg_angle = clear_bragg + self.bragg_offset
            self.order = self.crystal.find_order(energies.mean(), bragg_angle)

        self.mythen.raw_data = mythen_data
        self.mythen.pixel_limit_noise = pixel_limit_noise

        # Calculate the scale factor index vs energy_mono:
        # energy = k*i + energy[0]
        self.k = (energies[-1] - energies[0]) / float(len(energies) - 1)

        # Fit data with linear regression
        self.mythen.calc_linear_regression(threshold)

        # Fit data with 2D dispersion
        self.mythen.calc_dispersion_2d()

        # Fit data with 1D dispersion by using pixel projection
        self.mythen.calc_dispersion_1d(pixel_projection=True)

        # Fit data with 1D dispersion by using index projection
        self.mythen.calc_dispersion_1d(pixel_projection=False)

        # Use the 2D dispersion fitting as linear interpolation of the pixel
        # to energy equation.
        # energy = k*a pixel + k*.b + energy0
        self.a = self.k * self.mythen.fit_2d.a
        self.b = self.k * self.mythen.fit_2d.b + energies[0]

        # TODO use the 2D dispersion instead of 1D
        # Use the 1D dispersion fitting to calculate the e0 and set the p0,
        # energy and pixel with maximum intensity
        self.p0 = self.mythen.fit_1d_pixel.p0
        self.e0 = float(self.pixels2energies(self.p0))

        # Debug info
        self.log.debug('Fit linear: {}'.format(self.mythen.fit_lineal))
        self.log.debug('Fit 2D: {}'.format(self.mythen.fit_2d))
        self.log.debug('Fit 1D pixel: {}'.format(self.mythen.fit_1d_pixel))
        self.log.debug('Fit 1D index: {}'.format(self.mythen.fit_1d_index))
        self.log.info('Clear: e0={}, p0={}, a={}, b={}, k={}, '
                       'order={}'.format(self.e0, self.p0, self.a, self.b,
                                         self.k, self.order))
        self.log.debug('Exiting of elastic_line_calibration.')

    def get_energy_map(self, ceout_data, mask=None):
        '''
        Method to get the energy functions coefficients for each scan point.

        EnergyMap = A(p0-p) + B

             B**2 * pixel_size
        A =  ---------------------- * sqrt(B**2 - Chkl**2)
                4 * R * Chkl**2

        B = Ceout

        R: Radius of the Roland Circle
        Chkl: xxx

        :param ceout_data: np.array(floats)
        :return: [(A,B)]
        '''

        Chkl = self.crystal.getChkl(self.order)
        self.log.debug('Chkl: {}'.format(Chkl))
        R = self.rowland_rad
        pixels_size = self.mythen.pixels_size
        B = ceout_data.reshape((ceout_data.size, 1))
        A = (B**2 * pixels_size * np.sqrt(B**2 - Chkl**2)) / (4 * R * Chkl**2)
        self.log.debug('A: {}'.format(A))
        self.log.debug('B: {}'.format(B))
        p0 = self.p0
        p = np.array(range(self.mythen.nr_pixels))
        energy_map = A * (p0 - p) + B
        self.log.debug('Emap: {}'.format(energy_map))
        # TODO: Implement get_mask in mythen
        if mask is None:
            mask = self.mythen.mask
        energy_map_masked = np.ma.masked_array(energy_map)
        energy_map_masked.mask = mask
        return energy_map_masked

    def calc_mspectra(self, ceout_data, raw_mythen_data, dispersion_vector,
                      dispersion_threshold=0.1, roi=None, remove_noise=True):

        dispersion_vector = np.array(dispersion_vector)

        # Prepare mask with the bad channels and the roi
        bkp_roi = self.mythen.get_roi()
        self.mythen.raw_data = raw_mythen_data
        if roi is not None:
            self.mythen.set_roi(roi)
        mythen_mask = self.mythen.mask
        mythen_data = self.mythen.data
        if bkp_roi is None:
            self.mythen.remove_roi()
        else:
            self.mythen.set_roi(bkp_roi)

        # Concatenate the dispersion mask with the bad_channels and roi mask
        dispersion_mask = (dispersion_vector < dispersion_threshold)
        mask = mythen_mask & dispersion_mask
        mythen_data.mask = mask

        energy_map = self.get_energy_map(ceout_data, mask=mask)

        if remove_noise:
            mythen_data -= self.mythen.get_pixel_noise()

        mythen_data *= dispersion_vector

        intensity_vector = []

        # TODO: ask why
        emap = energy_map * 2

        for idx, energy in enumerate(ceout_data):
            if idx == 0:
                e_min = energy
            else:
                e_min = ceout_data[idx - 1]
            if idx == ceout_data.size - 1:
                e_max = energy
            else:
                e_max = ceout_data[idx + 1]

            e_idxs = (emap > (e_min + energy)) * (emap < (e_max + energy))
            intensity_vector.append(mythen_data[e_idxs].sum())






        # self.spectra = interp1d(ceout_real, Intensity, copy=False,
        #                         bounds_error=False, fill_value=0.0)
        #
        # return noise, pixels, ceout_real, np.array(Ns), np.array(
        #     NDisp_list), Emap.data, Disp, norm, np.array(
        #     Intensity0), self.mythen.data[:,
        #                  int(self.p0)] / self.mythen.dispersion_max, np.array(
        #     Intensity)
