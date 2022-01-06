import os
import sys
sys.path.insert(0, '../src')
import numpy as np
from shutil import rmtree

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


def test_parser():
    N = 100

    # generate fake data
    for _ in range(10):
        coordinates = np.random.uniform(0, 10, (N, 3))
        tcc.utility.dump_xyz('data.xyz', coordinates)
    box = [10, 10, 10]

    # run tcc, all result will be placed in a folder named 'tcc'
    parser = tcc.Parser('tcc')
    parser.run(
        'data.xyz', box=box,
        # all TCC parameters will go there
        rcutAA=1.8, clusts=True, raw=True, PBCs=False, analyse_all_clusters=True,
    )
    parser.parse()

    # output movie with cluster count
    xyz = tcc.XYZ('data.xyz', usecols=(1, 2, 3))
    tcc.utility.make_movie_count(
        xyz=xyz, tcc=parser,
        output="cluster-count.xyz",
        clusters=['sp3a', 'sp4a', 'sp5a'],  # choose clusters to dump to xyz file
    )
    parser.frame_bool(0)
    parser.frame_count(0)
    xyz.close()

    os.remove("cluster-count.xyz")
    os.remove("data.xyz")

    tcc_path_abs = os.path.abspath('tcc')

    # parse folder in different workiing dir
    os.chdir('..')
    new_parser = tcc.Parser(tcc_path_abs)
    new_parser.parse()
    new_parser.frame_bool(0)
    new_parser.frame_count(0)

    rmtree(tcc_path_abs)
