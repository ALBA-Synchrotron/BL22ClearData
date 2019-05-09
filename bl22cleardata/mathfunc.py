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

import numpy as np
from scipy.optimize import curve_fit
from .constants import M_RAW, IO, BAD_PIXEL


def get_mythen_data(data):
    """
    Method to extract the Mythen data from the raw data. It normalizes the
    Mythen data by the I0 and remove the bad channels.
    :param data: Dictionary with the raw data
    :type: dict
    :return: Mythen date pre-prosed
    :type: numpy.array
    """

    m_raw = np.array(data[M_RAW])
    i0 = np.array(data[IO])

    # Remove bad pixels
    m_raw = m_raw[:, range(BAD_PIXEL)]

    # Normalize the mythen data
    m_norm = (m_raw / i0[:, np.newaxis]) * i0.mean()
    return m_norm


def get_data_without_noise(data, roi_low, roi_high):
    # The original code use only the noise form 0 to a pixel_limit_noise 600.
    # Refactor to use the data out of the roi
    low_noise = data[:, 0:roi_low].mean()
    high_noise = data[:, roi_high:BAD_PIXEL].mean()
    noise = (low_noise + high_noise) / 2
    data_wn = data - noise
    return data_wn


def normalize(x, scale_min=0., scale_max=1.):
    """
    Normalize the signal by the MaxMin Scaler method
    Xmin = min(X)
    Xmax = max(X)
             X - Xmin
    Xnorm = -------------
             Xmax - Xmin

    Additional it is possible to change the scale
    Xnorm_scaled = (scale_max - scale_min) * Xnorm + scale_min

    :param x: Data vector
    :type: numpy.array
    :param scale_min: Minimum value of the new scale
    :type: float
    :param scale_max: Maximum value of the new scale
    :type: float

    :return: Normalized and rescaled data vector
    :type: numpy.array
    """
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max == x_min:
        return x
    x_norm = (x - x_min) / (x_max - x_min)

    x_rescaled = (scale_max - scale_min) * x_norm + scale_min

    return x_rescaled


def probability_distribution(x, order=2):
    """
    Calculate the probability distribution
    :param x: Data vector
    :type: numpy.array
    :param order:
    :type: int
    :return:
    """
    x = normalize(x, 0., 1.)
    x = x ** order
    x = x / x.sum()
    return x


def linear_regression(img, threshold=0.7, order=2):
    """
    Calculate the linear regression
    :param img: Image
    :type: numpy.array
    :param threshold: Min value threshold [0,1]
    :type: float
    :param order: Probability distribution order
    :return:
    """
    img_norm = normalize(img)
    img_clipped = img_norm.clip(min=threshold, max=1)
    pd_xy = probability_distribution(img_clipped, order=order)
    x_idx = np.arange(img.shape[1])
    y_idx = np.arange(img.shape[0])
    XY = y_idx.reshape((y_idx.size, 1)) * x_idx
    Px = pd_xy.sum(axis=0)  # marginal distribution of probability
    Py = pd_xy.sum(axis=1)  # marginal distribution of probability
    Xmean = (Px*x_idx).sum()  # calculation of <X>
    Ymean = (Py*y_idx).sum()  # calculation of <Y>
    X2mean = (Px*x_idx**2).sum()  # calculation of <X^2>
    Y2mean = (Py*y_idx**2).sum()  # calculation of <Y^2>
    XYmean = (pd_xy * XY).sum()  # calculation of <XY>
    a = (XYmean - Xmean*Ymean) / (X2mean - Xmean**2)
    b = Ymean - a*Xmean
    Xstd = np.sqrt(X2mean - Xmean**2)
    Ystd = np.sqrt(Y2mean - Ymean**2)
    return a, b, Xmean, Ymean, Xstd, Ystd


def calc_autoroi(m_data, noise_percent=2.5):
    m_norm = normalize(m_data)
    m_roi_x = m_norm.sum(axis=0)

    # Reduce noise
    noise_level = m_roi_x.max() * (noise_percent / 100)
    m_roi_x_clipped = m_roi_x.clip(min=noise_level)
    diff_m_roi = np.diff(np.diff(m_roi_x_clipped))
    diff_norm = normalize(diff_m_roi)
    r = np.where(diff_norm > diff_norm.mean()+1e-4)[0]
    auto_roi_low = int(r[0])
    auto_roi_high = int(r[-1])

    return auto_roi_low, auto_roi_high


def dispersion_2d(a, x0, y0, alpha, beta, X, Y):
    XX = X - x0
    YY = Y - y0
    arg = 0.5 * (((XX - YY / a) / alpha) ** 2) + 0.5 * ((YY / beta) ** 2)
    z = 1 / (2 * np.pi * alpha * beta) * np.exp(-arg)
    if z.sum() == 0:
        return 0
    return z / z.sum()


def get_statistics(x, p, order=2):
    p = probability_distribution(p, order=order)
    Xmean = (p * x).sum()
    Xstd = np.sqrt((p * x ** 2).sum() - Xmean ** 2)
    return Xmean, Xstd


def get_best_fit(x, y, func):
    x_mean, x_std = get_statistics(x, y, order=4)
    first_point = [y.mean(), x_mean, x_std]
    try:
        popt, pcov = curve_fit(func, x, y, p0=first_point)
        x_max, x_mean, x_std = popt
    except Exception:
        x_max, x_mean, x_std = first_point
    return x_max, x_mean, x_std


def gauss_function(x, a, x0, sigma):
    sigma = max(sigma, 1)
    return a*np.exp(-(x-x0)*(x-x0)/(2*sigma**2))