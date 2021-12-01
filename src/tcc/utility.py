import os
import numpy as np


def dump_xyz(filename, positions, comment=''):
    """
    Dump positions into an xyz file

    Args:
        filename (str): the name of the xyz file, it can be an existing file
        positions (numpy.ndarray): the positions of particles, shape (n, dim)
        comment (str): the content in the comment line

    Return:
        None
    """
    n, dim = positions.shape
    with open(filename, 'a') as f:
        np.savetxt(
            f, positions, delimiter=' ',
            header='%s\n%s' % (n, comment),
            comments='',
            fmt=['A %.8e'] + ['%.8e' for i in range(dim - 1)]
        )


def make_movie_bool(xyz, tcc, output, start=0, stop=None, clusters=None):
    if output in os.listdir('.'):
        os.remove(output)
    if isinstance(clusters, type(None)):
        clusters = tcc.cluster_bool.keys()
    if isinstance(stop, type(None)):
        stop = len(tcc)
    comment = "x y z " + " ".join(clusters)
    for f in range(start, stop):
        coord = xyz[f]  # n, 3
        is_cluster = tcc.frame_bool(f, clusters).to_numpy()  # n, m
        data = np.concatenate((coord, is_cluster), axis=1)  # (n, m + 3)
        dump_xyz(output, data, comment=comment)


def make_movie_count(xyz, tcc, output, start=0, stop=None, clusters=None):
    if output in os.listdir('.'):
        os.remove(output)
    if isinstance(clusters, type(None)):
        clusters = tcc.cluster_bool.keys()
    if isinstance(stop, type(None)):
        stop = len(tcc)
    comment = "x y z " + " ".join(clusters)
    for f in range(start, stop):
        coord = xyz[f]  # n, 3
        is_cluster = tcc.frame_count(f, clusters).to_numpy()  # n, m
        data = np.concatenate((coord, is_cluster), axis=1)  # (n, m + 3)
        dump_xyz(output, data, comment=comment)
