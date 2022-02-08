import tcc
import numpy as np

"""
Generate fake data
"""
N = 100
for _ in range(10):
    coordinates = np.random.uniform(0, 10, (N, 3))
    tcc.utility.dump_xyz('data.xyz', coordinates)
box = [10, 10, 10]

"""
run tcc, all result will be placed in a folder named 'tcc_result'
"""
parser = tcc.Parser('tcc_result')
parser.run(
    'data.xyz', box=box,
    # all TCC parameters will go there
    rcutAA=1.8, clusts=True, raw=True,
    PBCs=False, analyse_all_clusters=True,
)
parser.parse()

"""
output movie with cluster count (occupation)
"""
xyz = tcc.XYZ('data.xyz', usecols=(1, 2, 3))
tcc.utility.make_movie_count(
    xyz=xyz, tcc=parser,
    output="cluster-count.xyz",
    # the clusters to be saved to XYZ file
    clusters=['sp3a', 'sp4a', 'sp5a'],
)
