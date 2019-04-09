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

import sys
import argparse
import logging
import logging.config
from .__init__ import version
from .calibration import main as calibration_main
from .constants import BAD_PIXEL
from .spectra import main as spectra_main

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'simple': {
            'format': '%(levelname)-8s %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        # 'file': {
        #     'level': 'INFO',
        #     'class': 'logging.FileHandler',
        #     'filename': '',
        #     'mode': 'w',
        #     'encoding': 'utf-8',
        #     'formatter': 'verbose'
        # }
    },
    'loggers': {
        'Application': {
            'handlers': ['console'],
            'propagate': True,
            'level': 'INFO',
        },
        'bl22cleardata': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        }
    }
}


def end(log, err_no=0):
    # log_file = LOGGING_CONFIG['handlers']['file']['filename']
    # log.info('Log saved on: {0}'.format(log_file))
    for h in log.handlers:
        h.flush()
    sys.exit(err_no)


def get_parser():
    desc = 'Mythen PostProcessing scripts\n'
    desc += 'Version: {}.\n'.format(version)
    epi = 'Documentation: \n'
    epi += 'Copyright 2017 CELLS / ALBA Synchrotron, Bellaterra, Spain.'
    fmt = argparse.RawTextHelpFormatter
    parse = argparse.ArgumentParser(description=desc,
                                    formatter_class=fmt,
                                    epilog=epi)
    ver = '%(prog)s {0}'.format(version)
    parse.add_argument('--version', action='version', version=ver)
    subps = parse.add_subparsers(help='commands')

    # -------------------------------------------------------------------------
    #                           Calibration command
    # -------------------------------------------------------------------------
    calib_cmd = subps.add_parser('calib', help='Generate the mythen '
                                              'calibration files')
    calib_cmd.set_defaults(which='calib')
    calib_cmd.add_argument('scan_file', help='Spec file with the scan')
    calib_cmd.add_argument('scan_id', help='Scan number')
    # TODO: Analyze if there is a way to change it
    calib_cmd.add_argument('output_file', help='Output file name')
    calib_cmd.add_argument('--noise', default=2.5, type=float,
                           help='Noise percent to be removed')
    calib_cmd.add_argument('--threshold', default=0.7, type=float,
                           help='Threshold level [0, 1] use on the fit')
    calib_cmd.add_argument('--roi', default='[0, {}]'.format(BAD_PIXEL),
                           help='Set ROI e.g --roi=[400,900]')
    calib_cmd.add_argument('--no_auto_roi', action='store_false',
                           help='No use the auto_roi generation')
    calib_cmd.add_argument('-d', '--debug', action='store_true',
                           help='Activate log level DEBUG')

    # -------------------------------------------------------------------------
    #                           Spectra command
    # -------------------------------------------------------------------------
    spectra_cmd = subps.add_parser('spectra', help='Generate the ceout '
                                                   'spectra plot')
    spectra_cmd.set_defaults(which='spectra')
    spectra_cmd.add_argument('scan_file', help='Spec file with the scan')
    spectra_cmd.add_argument('scan_id', help='Scan number')
    spectra_cmd.add_argument('calib_file', help='Calibration json file')
    # TODO: Analyze if there is a way to change it
    spectra_cmd.add_argument('output_file', help='Output file name')
    spectra_cmd.add_argument('-d', '--debug', action='store_true',
                             help='Activate log level DEBUG')

    # -------------------------------------------------------------------------
    #                           IcePAP commands
    # -------------------------------------------------------------------------
    return parse


# def get_filename(host, command, filename='', log=False):
#     value = time.strftime('%Y%m%d_%H%m%S')
#     ext = ['cfg', 'log'][log]
#     if host == '':
#         host = 'icepap'
#     new_filename = '{0}_{1}_{2}.{3}'.format(value, host, command, ext)
#     if log or filename == '':
#         filename = new_filename
#     return os.path.abspath(filename)

def config_loggers(debug):
    verbose = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    simple = '%(levelname)-8s %(message)s'

    log_ch = logging.StreamHandler()
    if debug:
        level = logging.DEBUG
        fmt = logging.Formatter(verbose)
    else:
        level = logging.INFO
        fmt = logging.Formatter(simple)

    log_ch.setLevel(level)
    log_ch.setFormatter(fmt)

    # Application Logger
    log_app = logging.getLogger('app')
    log_app.setLevel(level)
    log_app.addHandler(log_ch)

    # Library
    log_lib = logging.getLogger('bl22cleardata')
    log_lib.setLevel(level)
    log_lib.addHandler(log_ch)


def main():
    parser = get_parser()
    args = parser.parse_args()
    if len(sys.argv) < 2:
        args = parser.parse_args(['-h'])
    config_loggers(args.debug)
    log = logging.getLogger('app')

    # Calibration Command
    if args.which == 'calib':
        log.info('Running calibration...')
        try:
            roi = eval(args.roi)
            calibration_main(scan_file=args.scan_file,
                             scan_id=args.scan_id,
                             output_file=args.output_file,
                             auto_roi=args.no_auto_roi,
                             user_roi=roi,
                             threshold=args.threshold,
                             noise_percent=args.noise)

        except Exception as e:
            log.error('The calibration failed: {}'.format(e))
            end(log, -1)

    # Spectra Command
    if args.which == 'spectra':
        log.info('Running spectra calculation...')
        try:
            spectra_main(scan_file=args.scan_file,
                         scan_id=args.scan_id,
                         calib_file=args.calib_file,
                         output_file=args.output_file)
        except Exception as e:
            log.error('The spectra calculation failed: {}'.format(e))
            end(log, -1)


if __name__ == '__main__':
    main()
