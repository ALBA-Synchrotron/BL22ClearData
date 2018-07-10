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
        self.mythen.calc_dispersion_1d(axis_x=True)

        # Fit data with 1D dispersion by using index projection
        self.mythen.calc_dispersion_1d(axis_x=False)

        # Use the 2D dispersion fitting as linear interpolation of the pixel
        # to energy equation.
        # energy = k*a pixel + k*.b + energy0
        self.a = self.k * self.mythen.fit_2d.a
        self.b = self.k * self.mythen.fit_2d.b + energies[0]

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

