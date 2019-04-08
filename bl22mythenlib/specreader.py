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

import logging
import os


def get_filename(filename, suffix='out', len_auto_index=3):
    fname, fext = os.path.splitext(filename)
    auto_index = 0
    while len_auto_index > 0:
        filename = '{0}_{1}_{2:0{3}d}{4}'.format(fname, suffix, auto_index,
                                                 len_auto_index, fext)
        if not os.path.exists(filename):
            break
        auto_index += 1

    return filename


def read_raw_data_spec(log, file_obj, scan_id):
    def print_log(msg):
        if log.isEnabledFor(logging.DEBUG):
            log_msg_header = 'ReadRawDataSpec: {0}'
            log.debug(log_msg_header.format(msg))

    found = False
    print_log('Finding scan {0} on {1}'.format(scan_id, file_obj.name))
    for line in file_obj:
        line_lower = line.lower()
        start_scan = '#s {0}'.format(scan_id)
        if start_scan in line_lower:
            found = True
            break

    if not found:
        raise RuntimeError('The scan {0} is not in '
                           'the file'.format(scan_id, file_obj))

    snapshots = {}
    data = {}
    snapshots_names = []
    snapshots_values = []
    channels_names = []
    channel_1d = ''

    # Skip header
    print_log('Skipping header')
    while True:
        line = file_obj.readline()
        line_lower = line.lower()
        if '#o' in line_lower or '#l' in line_lower:
            break

    # Read snapshots channels names
    print_log('Reading snapshots')
    while True:
        if '#o' not in line.lower():
            break
        snapshots_names += line.split()[1:]
        line = file_obj.readline()

    # Read snapshots channels values
    while True:
        if '#p' not in line.lower():
            break
        snapshots_values += list(map(float, line.split()[1:]))
        line = file_obj.readline()

    if len(snapshots_names) > 0:
        if len(snapshots_names) != len(snapshots_values):
            print_log('There is a mismatch on the snapshots name and value '
                      'length.')
        else:
            for name, value in zip(snapshots_names, snapshots_values):
                snapshots[name] = [value]

    while True:
        line_lower = line.lower()
        if '#l' in line_lower:
            break
        if '#@' in line_lower:
            if 'det' in line_lower:
                channel_1d = line.split()[1]
                data[channel_1d] = []
        line = file_obj.readline()

    # Read channels name
    print_log('Reading channels names')
    channels_names += line.split()[1:]
    for name in channels_names:
        data[name] = []

    # Read channels data
    print_log('Reading channels data')
    while True:
        line = file_obj.readline()
        if '#' in line:
            break
        if '@a' in line.lower():
            # Read 1d data
            ch1d_data = list(map(float, line.split()[1:]))
            data[channel_1d].append(ch1d_data)
        else:
            channels_data = list(map(float, line.split()))
            for name, value in zip(channels_names, channels_data):
                data[name].append(value)

    return data, snapshots


def read_scan(filename, scan_id, log=None):
    if log is None:
        log = logging.getLogger('bl22mythenlib.read_scan')
    with open(filename, 'r') as f:
        return read_raw_data_spec(log, f, scan_id)


def concatenate_scans(filename, scans_ids, log=None):
    pass
