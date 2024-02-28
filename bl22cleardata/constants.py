# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# This file is part of BL22 Mythen Post Processing ()
#
# Author(s): Dominique Heinis <dheinis@cells.es>,
#            Roberto J. Homs Puron <rhoms@cells.es>
#
# Copyright 2008 CELLS / ALBA Synchrotron, Bellaterra, Spain
#
# Distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
#
# You should have received a copy of the GNU General Public License
# along with the software. If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

#from crystal import CrystalGe, CrystalSi


# Beamline Channels names
M_RAW = 'm_raw'
ENERGY = 'energyc'
CBRAGG = 'clear_bragg'
CEOUT = 'ceout'
CAZ = 'caz'

# Mythen
PIXEL_SIZE = 50e-6
BAD_PIXEL = 1000


# Analyzer selector
def get_crystal(caz_pos):
    return 1
    #return CrystalSi([1, 1, 1])
    # TODO: Implement selection by position
    # if -183 < caz_pos < -180:
    #     return CrystalSi([1, 1, 1])
    # else:
    #     raise RuntimeError('It is not a valid position for a crystal')
