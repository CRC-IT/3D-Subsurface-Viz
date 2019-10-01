"""
KDTree search

>>> df = pd.read_csv('data/points.csv', low_memory=False)
>>> tree = build(df, debug=False)
>>> query(tree, 'C201_00', 100, debug=False)
['C201_00', 'C202_00', 'C224_00', 'C224_01']
"""
import numpy as np
import pandas as pd
import scipy.spatial as sp
import time


def build(df, xyz=['MAP_EASTING', 'MAP_NORTHING', 'TVDSS'], z_prune=-2000,
          debug=True):
    """Build the KDTree and prune off upper bore, must pass back pruned df"""
    df_ = df[df[xyz[-1]] <= z_prune]
    start = time.time()
    tree_ = sp.cKDTree(df_[xyz])
    if debug:
        print("Build time: {:0.3f}".format(time.time() - start))
    return tree_, df_


def query(tree, name, radius, xyz=['MAP_EASTING', 'MAP_NORTHING', 'TVDSS'],
          label='NEW_WELL_NAME', debug=True):
    """Find neighboring points along a wellbore and distill to well names"""
    tree_, df_ = tree
    points = df_[df_[label] == name][xyz]
    start = time.time()
    listoflists = tree_.query_ball_point(points, radius)
    indexes = np.concatenate(listoflists)
    names = sorted(df_.iloc[indexes][label].unique())
    if debug:
        print("Query time: {:0.3f}".format(time.time() - start))
        print("Points queried: %s, Points returned: %s, Wells returned: %s" %
              (len(listoflists), len(indexes), len(names)))
    return names


if __name__ == '__main__':
    import doctest
    doctest.testmod()

# p = (4246424, 4017708, -2344)  # C201 should return C344 C202...
