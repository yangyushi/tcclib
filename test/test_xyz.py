import os
import numpy as np
import sys
sys.path.insert(0, '../src')
import tcc

FILENAME = 'temp.xyz'

def test_xyz():
    if FILENAME in os.listdir('.'):
        os.remove(FILENAME)
    n_frame = 10

    configurations = []
    for f in range(n_frame):
        num = np.random.randint(100, 200)
        coord = np.random.uniform(0, 10, (num, 3))
        configurations.append(coord)
        tcc.dump_xyz(FILENAME, coord, comment="random")

    xyz = tcc.XYZ(FILENAME, usecols=(1, 2, 3), align_opt=True)
    for f, frame in enumerate(xyz):
        assert np.allclose(frame, configurations[f])

    xyz.close()
    if FILENAME in os.listdir('.'):
        os.remove(FILENAME)

if __name__ == "__main__":
    test_xyz()
