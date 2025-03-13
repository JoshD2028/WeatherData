import h5py
import numpy as np
import pandas as pd


def load_h5_file(filename):
    '''gets the file name and returns a dict of dataframes containing the data
    '''
    f = h5py.File(filename, 'r')
    data = {k: (np.array(f[k])) for k in list(f.keys())}
    for k, v in data.items():
        index = pd.MultiIndex.from_product([range(s) for s in v.shape], names=[
                                           f'x{i}' for i in range(v.ndim)])
        data[k] = pd.DataFrame({'A': v.flatten()}, index=index).reset_index()
    return data
