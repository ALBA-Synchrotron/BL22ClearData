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
from scipy.optimize import fmin
from .utils import linear_regression, gauss_function, get_best_fit


class DataFitted(object):
    """
    Class to store the fitted values
    """
    def __init__(self, a=None, b=None, p0=None, i0=None, p0_std=None,
                 i0_std=None, disp_exp=None, disp_theo=None, disp_max=None):
        self.a = a
        self.b = b
        self.p0 = p0
        self.i0 = i0
        self.p0_std = p0_std
        self.i0_std = i0_std
        self.disp_exp = disp_exp
        self.disp_theo = disp_theo
        self.disp_max = disp_max

    def __repr__(self):
        value = 'a={}, b={}, p0={}, i0={}, p0_std={}, i0_std={}, ' \
                'disp_exp={}, disp_theo={}, ' \
                'disp_max={}'.format(self.a, self.b, self.p0, self.i0,
                                     self.p0_std, self.i0_std, self.disp_exp,
                                     self.disp_theo, self.disp_max)
        return value


class Mythen(object):
    """
    Class for the Mythen 1K detector
    """
    nr_pixels = 1280
    pixels_size = 50e-6

    def __init__(self, data=None, dead_pixels=None,
                 disp_func=gauss_function):
        self._dead_pixels = None
        self._rois = [[0, self.nr_pixels]]
        self._pixels_masked = None
        self._data = None
        self._data_masked = None

        self.pixel_limit_noise = 600
        self.fit_lineal = DataFitted()
        self.fit_2d = DataFitted()
        self.fit_1d_pixel = DataFitted()
        self.fit_1d_index = DataFitted()
        if data is not None:
            self.raw_data = data
        self.disp_func = disp_func
        self.dead_pixels = dead_pixels

    def _update_bad_channels(self):
        idx = np.arange(1, self.nr_pixels + 1)
        mask = np.zeros(self.nr_pixels, dtype="bool")
        mask[self._dead_pixels - 1] = True
        mask[:self.pixel_limit_noise] = [True] * self.pixel_limit_noise
        self._pixels_masked = np.ma.masked_array(idx, mask=mask)

    @property
    def dead_pixels(self):
        return self._dead_pixels

    @dead_pixels.setter
    def dead_pixels(self, value):
        if value is None:
            self._dead_pixels = np.arange(1100, self.nr_pixels + 1)
        else:
            self._dead_pixels = np.array(value)
        self._update_bad_channels()

    @property
    def data(self):
        return self._data_masked

    @property
    def raw_data(self):
        return self._data

    @raw_data.setter
    def raw_data(self, value):
        self._data = np.array(value)
        if len(self._data.shape) > 1:
            mask = np.stack([self._pixels_masked.mask] * self._data.shape[0])
        else:
            mask = self._pixels_masked.mask
        self._data_masked = np.ma.masked_array(self._data, mask=mask)

    def add_roi(self, pmin=500, pmax=1100):
        """
        Add roi do the rois list
        :param pmin: int
        :param pmax: int
        :return: int -> ROI id
        """
        if pmin < 1 or pmin > self.nr_pixels or pmax < 1 \
                or pmax > self.nr_pixels:
            raise ValueError('pmin and pmax should be between '
                             '[1,{0}]'.format(self.nr_pixels))
        pmin = int(min(pmin, pmax))
        pmax = int(max(pmin, pmax))
        self._rois.append([pmin, pmax])
        return len(self._rois)

    def get_rois(self):
        return list(self._rois)

    def get_data(self, compressed=False, remove_noise=False, clip=False):
        """
        Method to applied some corrections to the raw data.
        :param compressed: bool -> Give only the data no-masked
        :param remove_noise: bool -> Remove the noise
        :param clip: bool
        :return: np.array
        """
        if compressed:
            colums = self._pixels_masked.compressed()
            data_shape = self._data_masked.shape
            data = self._data_masked.compressed().reshape(data_shape[0],
                                                          len(colums))
        else:
            colums = self.nr_pixels
            data = self._data_masked

        if remove_noise:
            data = data - self.get_pixel_noise()
        if clip:
            data = data.clip(min=0)
        rows = np.arange(self.raw_data.shape[0])
        return rows, colums, data

    def calc_linear_regression(self, threshold=0.7, order=2):
        """
        Calculate the linear regression parameters of the image. The
        parameters are saved on fit_linal variable.
        :return:
        """
        # a and b  are calculated respect to raw and column starting at
        # index 0.
        a, b, p0, i0, p0_std, i0_std = linear_regression(self.data,
                                                         threshold, order)
        # we have to shift to compensate pixel number 1 at index 0
        # y = ax + b --> y = a(x'-1) + b <==> ax' -a + b
        b = b - a

        self.fit_lineal = DataFitted(a, b, p0, i0, p0_std, i0_std)

    def calc_dispersion_2d(self):
        """
        Method to calculate the 2D dispersion fitting.
        :return:
        """

        rows, colums, data = self.get_data(compressed=True, remove_noise=True)
        X, Y = np.meshgrid(colums, rows)
        data_sum = data.sum()
        data /= data_sum
        a = self.fit_lineal.a
        p0 = self.fit_lineal.p0
        i0 = self.fit_lineal.i0
        p0_std = self.fit_lineal.p0_std
        i0_std = self.fit_lineal.i0_std

        cost = lambda v: ((self.dispersion_2d(a, p0, i0, p0_std, i0_std, X,
                                              Y) - data) ** 2).sum()
        new_a, new_p0, new_i0, new_p0_std, new_i0_std = fmin(cost,
                                                             [a, p0, i0,
                                                              p0_std,
                                                              i0_std])
        new_b = i0 - new_a * new_p0
        if np.abs((new_a - a) / a) < 0.1:
            self.fit_2d = DataFitted(new_a, new_b, new_p0, new_i0, new_p0_std,
                                     new_i0_std)

    def get_pixel_noise(self):
        noise = self.raw_data[:, 0:self.pixel_limit_noise]
        noise = noise.mean()
        return noise

    @staticmethod
    def dispersion_2d(a, x0, y0, alpha, beta, X, Y):
        XX = X - x0
        YY = Y - y0
        arg = 0.5 * (((XX - YY / a) / alpha) ** 2) + 0.5 * ((YY / beta) ** 2)
        z = 1 / (2 * np.pi * alpha * beta) * np.exp(-arg)
        return z / z.sum()

    def calc_dispersion_1d(self, axis_x=True):
    def calc_dispersion_1d(self, pixel_projection=True):
        """
        Metho to calculate the 1D dispersion fitting
        :param axis_x:
        :return:
        """
        rows, colums, data = self.get_data(compressed=True, remove_noise=True)
        axis = [1, 0][pixel_projection]
        x = [rows, colums][pixel_projection]
        disp_exp = data.max(axis=axis)

        x_max, x_mean, x_std = get_best_fit(x, disp_exp, self.disp_func)
        disp_theo = self.disp_func(x, x_max, x_mean, x_std)
        if pixel_projection:
            # Pixel information
            self.fit_1d_pixel = DataFitted(p0=x_mean, p0_std=x_std,
                                           disp_exp=disp_exp,
                                           disp_theo=disp_theo,
                                           disp_max=x_max)

        else:
            self.fit_1d_index = DataFitted(i0=x_mean, i0_std=x_std,
                                           disp_exp=disp_exp,
                                           disp_theo=disp_theo,
                                           disp_max=x_max)

    # def getResolutionParameters(self, a, b, axis="X"):
    #     I, pixels, norm = self.getData(compressed=True, removeNoise=True,
    #                                    clip=True)
    #     if axis == "X":
    #         X = pixels
    #         C = std.getLinearCompensation(norm, a, b, offset=self.p0, axis=1)
    #         R = C.mean(axis=0)
    #     else:
    #         X = I
    #         C = std.getLinearCompensation(norm, a, b, offset=self.i0, axis=0)
    #         R = C.mean(axis=1)
    #     # R = R - R.min()
    #     func = getattr(std, self.resolution_curve)
    #     Rmax, Rmean, Rstd = std.getBestFit(X, R, func)
    #     return Rmax, Rmean, Rstd, C, R
    #
    # def setDispersion(self, dispersion_curve="gauss_function",
    #                   dispersion_max=1, dispersion_mean=1.,
    #                   dispersion_sigma=1.):
    #     # be carefull when dispersion is experimental
    #     # dispersion_max == threshold
    #     # dispersion_sigma == gaussian filter sigma
    #     self.dispersion_curve = dispersion_curve
    #     self.dispersion_max = dispersion_max
    #     self.dispersion_mean = dispersion_mean
    #     self.dispersion_sigma = dispersion_sigma
    #     if self.dispersion_curve == "experimental":
    #         X = self.nr_pixels.compressed()
    #         norm = std.getNoiseLessSignal(self.norm,
    #                                       threshold=self.dispersion_max)
    #         D = (norm.mean(axis=0).compressed() - self.get_pixel_noise()).clip(
    #             min=0)
    #         Y = gaussian_filter(D, self.dispersion_sigma)
    #         self.dispersion_experimental = interp1d(X, Y, copy=True,
    #                                                 bounds_error=False,
    #                                                 fill_value=0.0)
    #
    # def getDispersion(self):
    #     return self.dispersion_curve, self.dispersion_max, self.dispersion_mean, self.dispersion_sigma
    #
    # def getDispersionFunction(self):
    #     if self.dispersion_curve == "gauss_function":
    #         return lambda x: std.gauss_function(x, self.dispersion_max,
    #                                             self.dispersion_mean,
    #                                             self.dispersion_sigma)
    #     elif self.dispersion_curve == "gauss2_function":
    #         return lambda x: std.gauss2_function(x, self.dispersion_max,
    #                                              self.dispersion_mean,
    #                                              self.dispersion_sigma)
    #     elif self.dispersion_curve == "gaussN_function":
    #         return lambda x: std.gaussN_function(x, self.dispersion_max,
    #                                              self.dispersion_mean,
    #                                              self.dispersion_sigma)
    #     elif self.dispersion_curve == "experimental":
    #         return self.dispersion_experimental
    #     else:
    #         pass
    #
    # def setResolution(self, resolution_curve="gauss_function",
    #                   resolution_max=1, resolution_mean=1.,
    #                   resolution_sigma=1.):
    #     self.resolution_curve = resolution_curve
    #     self.resolution_max = resolution_max
    #     self.resolution_mean = resolution_mean
    #     self.resolution_sigma = resolution_sigma
    #
    # def getResolution(self):
    #     return self.resolution_curve, self.resolution_max, self.resolution_mean, self.resolution_sigma
    #
    # def getResolutionFunction(self):
    #     if self.resolution_curve == "gauss_function":
    #         return lambda x: std.gauss_function(x, self.resolution_max,
    #                                             self.resolution_mean,
    #                                             self.resolution_sigma)
    #     elif self.resolution_curve == "gauss2_function":
    #         return lambda x: std.gauss2_function(x, self.resolution_max,
    #                                              self.resolution_mean,
    #                                              self.resolution_sigma)
    #     elif self.resolution_curve == "gaussN_function":
    #         return lambda x: std.gaussN_function(x, self.resolution_max,
    #                                              self.resolution_mean,
    #                                              self.resolution_sigma)
    #     elif self.resolution_curve == "experimental":
    #         return self.resolution_experimental
    #     else:
    #         pass