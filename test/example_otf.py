import tcc
import numpy as np


"""
Generate ideal gas (random points)
Different frames may have different numbers
"""
n_frame = 10
configurations = []
for f in range(n_frame):
    num = np.random.randint(1000, 1100)
    coord = np.random.uniform(0, 25, (num, 3))
    configurations.append(coord)

"""
Find clusters in the gas
"""
otf = tcc.OTF()
box = [25, 25, 25]
otf(
    configurations, box,
    # TCC parameters
    rcutAA=1.8, clusts=True, raw=True,
    PBCs=True, analyse_all_clusters=True,
)


"""
Output the coordinates with tcc clusters
The output file can be loaded in Ovito for rendering
"""
output = "result.xyz"  # the name of the output file
clusters = ["sp4a", "sp5c", "FCC"]  # name of the clusters you want to dump

for f, coord in enumerate(configurations):
    comment = "x y z " + " ".join(clusters)
    is_cluster = otf.frame_bool(f, clusters).to_numpy()
    data = np.concatenate((coord, is_cluster), axis=1)
    tcc.dump_xyz(output, data, comment=comment)
