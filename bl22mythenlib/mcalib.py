import numpy as np
from clear import Clear
from utils import read_raw_data_spec


def mcalib(filename, scan_id, ftype='spec', threshold=0.7,
           pixel_limit_noise=600):
    clear = Clear()

    if ftype == 'spec':
        data, snapshots = read_raw_data_spec(filename, scan_id)
    else:
        raise ValueError('Only read spec files')

    raw_mythen = np.array(data['m_raw'])
    energies = np.array(data['energyc'])
    i0 = np.array(data['n_i0_1'])
    cbragg = snapshots['clear_bragg']

    # Normalize the data
    mythen_norm = raw_mythen/(i0[:, np.newaxis])*i0.mean()

    clear.elastic_line_calibration(mythen_norm, energies, cbragg,
                                   threshold=threshold,
                                   pixel_limit_noise=pixel_limit_noise)
    return clear
