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


def normalize(data, b_min=0., b_max=1.):
    np.array(data)
    a_min = data.min()
    a_max = data.max()
    if a_max - a_min != 0:
        data = ((b_max - b_min) / float(a_max - a_min) * (data - a_min)) + \
                 b_min
    return data


def probability_distribution(data, order=2):
    data = normalize(data, 0., 1.)
    data = data ** order
    data = data / data.sum()
    return data


def linear_regression(data, threshold, order=2):
    p_xy = normalize(data).clip(min=threshold, max=1)
    p_xy = probability_distribution(p_xy, order=order)
    X = np.arange(data.shape[1])
    Y = np.arange(data.shape[0])
    XY = Y.reshape((Y.size, 1)) * X
    Px = p_xy.sum(axis=0)  # marginal distribution of probability
    Py = p_xy.sum(axis=1)  # marginal distribution of probability
    Xmean = (Px*X).sum()  # calculation of <X>
    Ymean = (Py*Y).sum()  # calculation of <Y>
    X2mean = (Px*X**2).sum()  # calculation of <X^2>
    Y2mean = (Py*Y**2).sum()  # calculation of <Y^2>
    XYmean = (p_xy * XY).sum()  # calculation of <XY>
    a = (XYmean - Xmean*Ymean) / (X2mean - Xmean**2)
    b = Ymean - a*Xmean
    Xstd = np.sqrt(X2mean - Xmean**2)
    Ystd = np.sqrt(Y2mean - Ymean**2)
    return a, b, Xmean, Ymean, Xstd, Ystd


def gauss_function(x, a, x0, sigma):
    sigma = max(sigma, 1)
    return a*np.exp(-(x-x0)*(x-x0)/(2*sigma**2))


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
    except:
        x_max, x_mean, x_std = first_point
    return x_max, x_mean, x_std





