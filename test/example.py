"""
This file shows a mimimum example of using tcclib
"""
import tcc
import numpy as np

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
