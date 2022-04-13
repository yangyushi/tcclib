"""
Getting coordinates of individual clusters with tcclib
"""
import tcc
import numpy as np
import matplotlib.pyplot as plt


"""
Generate ideal gas (random points)
Different frames may have different numbers
"""
n_frame = 10
configurations = []
for f in range(n_frame):
    num = np.random.randint(1500, 1600)
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
    PBCs=False, analyse_all_clusters=True,
)


"""
obtaining each clusters
otf == parser
"""
clusters = []
cluster_name = 'sp5a'
frame_num = 0
indices_of_clusteres = otf.cluster_detail[cluster_name][frame_num]
for indices in indices_of_clusteres:  # iter over indices for each cluster
    positions = configurations[frame_num][indices]
    clusters.append(positions)


"""
plot
"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for clust in clusters:
    ring = np.vstack((clust, clust[0][None, :]))
    ax.scatter(*clust.T)
    ax.plot(*ring.T, color='k')
plt.tight_layout()
plt.show()
