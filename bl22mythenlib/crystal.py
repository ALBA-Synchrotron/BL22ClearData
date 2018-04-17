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
        d = self._a/ np.linalg.norm(self._hkl * n)
        return float(d)

    def get_bragg(self, energy, n=1):
        """
        Calculate the bragg angle for a certain energy and order.
        :param energy: float -> Energy in eV
        :param n: int -> Order
        :return: float
        """
        d = self.get_distance(n)
        wavelength = self.hc * energy
        sin_bragg = wavelength / (2 * d)
        if np.abs(sin_bragg) >= 1:
            bragg = 90.0
        else:
            bragg = np.rad2deg(np.arcsin(sin_bragg)).item()
        return bragg

    def get_energy(self, bragg, n=1):
        """
        Calculate the energy in eV of certain bragg and order.
        :param bragg: float
        :param n: int
        :return: float
        """
        d = self.get_distance(n)
        wavelength = 2 * d * np.sin(np.deg2rad(bragg))
        energy = wavelength / self.hc
        return energy

    def find_order(self, energy, bragg_zero):
        """
        Find the order used to a certain energy and initial bragg angle.
        :param energy:
        :param bragg_zero:
        :return:
        """
        d = self.get_distance(1)
        wavelength = self.hc * energy
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
    Class for the .
    """
    def __init__(self, hkl):
        a = 0.5657900e-9
        BaseCrystal.__init__(self, a, hkl)



class Crystal(object):
    """
    Class to store the crystal hkl values and 'a' constant in meters.
    """

    si111 = {'a': 0.5430710e-9, 'h': 1, 'k': 1, 'l': 1}
    si220 = {'a': 0.5430710e-9, 'h': 2, 'k': 2, 'l': 0}
    si400 = {'a': 0.5430710e-9, 'h': 4, 'k': 0, 'l': 0}
    ge111 = {'a': 0.5657900e-9, 'h': 1, 'k': 1, 'l': 1}
    hc = 0.00123984193e-3  # [eV*m]

    def __init__(self, crystal='Si111', n=1):
        """
        Constructor of the class
        :param crystal: str
        :param n: int
        """

        self._crystal_name = None
        self._crystal_values = None
        self._n = None

        self.crystal = crystal
        self.order = n

    @property
    def crystal(self):
        """
        Get the crystal used
        :param crystal: str
        :return: None
        """
        return self._crystal_name

    @crystal.setter
    def crystal(self, value):
        """
        Set the crystal used
        :param value: str
        :return: None
        """
        self._crystal_name = value
        self._crystal_values = self.__getattribute__(value.lower())

    @property
    def order(self):
        """
        Get the crystal order
        :return: int
        """
        return self._n

    @order.setter
    def order(self, value):
        """
        Set the crystal order
        :param value: int
        :return: None
        """
        self._n = value

    @property
    def distance(self):
        """
        Get the distance between atomic planes in meters
            d=a/sqrt(h^2+k^2+l^2) for a cubic crystal.
        :return: float
        """
        aa = self._crystal_values['a']
        hh = self._crystal_values['h']
        kk = self._crystal_values['k']
        ll = self._crystal_values['l']
        d = aa / math.sqrt(hh**2 + kk**2 + ll**2)
        return d

    def get_energy(self, bragg):
        """
        Calculate the energy[eV] for a determine bragg angle[degrees]
           energy = hc/wavelength
           n*wavelength=2*d*sin(bragg).
           'a' is the lattice spacing (of 0.5430710nm in the case of Silicon).
           'd' is the distance between planes of crystalline structure.

        :param bragg: float
        :return: float
        """

        bragg_rad = deg2rad(bragg)
        d = self.distance
        wavelength = 2 * d * math.sin(bragg_rad) / self._n
        energy = self.hc / wavelength
        return energy

    def get_bragg(self, energy):
        """
        Calculate the bragg angle[degrees] for a determine energy[eV]
           energy = hc/wavelength
           n*wavelength=2*d*sin(bragg).
           Where 'd' is the distance between atomic planes.
           We also have: d=a/sqrt(h^2+k^2+l^2) for a cubic crystal.
           'a' is the lattice spacing (of 0.5430710nm in the case of Silicon).
           'd' is the distance between planes of crystalline structure.

        :param energy: float
        :return: float
        """
        wavelength = self.hc / energy
        d = self.distance
        value = wavelength / (2 * d)
        if value > 1 or value < -1:
            msg = 'It is not possible to calculate bragg for this energy: ' \
                  '{0} eV with the crystal set {1}'.format(energy,
                                                           self.crystal)
            raise ValueError(msg)

        bragg_rad = math.asin(value)
        bragg = rad2deg(bragg_rad)
        return bragg

