import numpy as np
from clear import Clear
from utils import read_raw_data_spec, CBRAGG, M_RAW, ENERGY, IO


def mcalib(filename, scan_id, threshold=0.7,
           pixel_limit_noise=600, ftype='spec'):
    clear = Clear()

    if ftype == 'spec':
        print 'in spec'
        data, snapshots = read_raw_data_spec(filename, [scan_id])[scan_id]
    else:
        raise ValueError('Only read spec files')

    raw_mythen = np.array(data[M_RAW])
    energies = np.array(data[ENERGY])
    i0 = np.array(data[IO])
    cbragg = snapshots[CBRAGG]

    # Normalize the data
    mythen_norm = raw_mythen/(i0[:, np.newaxis])*i0.mean()

    clear.elastic_line_calibration(mythen_norm, energies, cbragg,
                                   threshold=threshold,
                                   pixel_limit_noise=pixel_limit_noise)
    return clear
