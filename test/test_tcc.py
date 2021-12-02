import numpy as np
import sys
sys.path.insert(0, '../src')
import tcc


def test_otf():
    n_frame = 4

    configurations = []
    for f in range(n_frame):
        num = np.random.randint(10000, 11000)
        coord = np.random.uniform(0, 100, (num, 3))
        configurations.append(coord)

    tcc_calc = tcc.OTF()
    box = [100, 100, 100]
    tcc_calc(configurations, box)

    for f in range(n_frame):
        assert tcc_calc.frame_bool(f).shape[0] == configurations[f].shape[0]
        assert tcc_calc.frame_count(f).shape[0] == configurations[f].shape[0]
        assert len(tcc_calc.population) == n_frame
