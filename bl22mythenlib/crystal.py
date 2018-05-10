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


class BaseCrystal(object):
    hc = 0.00123984193e-3 # [eV*m]

    def __init__(self, a, hkl):
        """
        Basic Cystal class
        :param a: float -> Lattice constant for the element
        :param hkl: [int] -> Miller number
        """
        self._a = a
        self._hkl = np.array(hkl)

    def get_distance(self, n=1):
        """
        Get the reticular distance between atomic planes in meters
            d=a/sqrt((h^2+k^2+l^2)*n) for a cubic crystal.
            a: Material lattice
            hkl: Miller numbers
            n: order
        :param n: int
        :return: float
        """
        d = self._a / np.linalg.norm(self._hkl * n)
        return float(d)

    def get_braggs(self, energies, n=1):
        """
        Calculate the bragg angle for a certain energy and order.
        :param energies: [float] -> Energy in eV
        :param n: int -> Order
        :return: buffer -> Pointer to the array buffer
        """
        d = self.get_distance(n)
        wavelength = self.hc * np.array(energies)
        sin_bragg = wavelength / (2 * d)
        bragg = np.rad2deg(np.arcsin(sin_bragg))
        np.place(bragg, np.isnan(bragg), 90)

        return np.getbuffer(bragg)

    def get_energies(self, braggs, n=1):
        """
        Calculate the energy in eV of certain bragg and order.
        :param braggs: [float]
        :param n: int
        :return: buffer -> Pointer to the array buffer.
        """
        d = self.get_distance(n)
        wavelength = 2 * d * np.sin(np.deg2rad(braggs))
        energy = wavelength / self.hc
        return np.getbuffer(energy)

    def find_order(self, energies, bragg_zero):
        """
        Find the order used to a certain energy and initial bragg angle.
        :param energies:
        :param bragg_zero:
        :return:
        """
        d = self.get_distance(1)
        wavelength = self.hc / energies
        orders = np.arange(1, np.floor(2 * d / wavelength) + 1)
        braggs = np.rad2deg(np.arcsin(wavelength / (2. * d) * orders))
        deltas_braggs = np.abs(braggs - bragg_zero)
        idx = np.argmin(deltas_braggs)
        n = orders[idx]
        return n

    # TODO: Implement
    # def getChkl(self, order):
    #     index = order * self.cut
    #     Q = self.getQ(index)
    #     C = np.linalg.norm(Q) * std.hc / 2.
    #     return C / (2. * np.pi)
    #


class CrystalSi(BaseCrystal):
    """
    Class for the Silicon.
    """
    def __init__(self, hkl):
        a = 0.5430710e-9
        BaseCrystal.__init__(self, a, hkl)


class CrystalGe(BaseCrystal):
    """
    Class for the Germanium.
    """
    def __init__(self, hkl):
        a = 0.5657900e-9
        BaseCrystal.__init__(self, a, hkl)


