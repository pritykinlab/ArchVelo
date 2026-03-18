import numpy as np
import multivelo as mv
import scanpy as sc
import pandas as pd
import scipy


def get_types(peaks, peaks_to_genes):
    ann_peaks = []
    for peak in peaks: 
        cur_ann = peaks_to_genes.loc[peak]
        peak_type = cur_ann['type']
        ann_peaks.append(peak_type)
    return ann_peaks

def collapse_types_func(arr):
    arr = arr.T
    arr['type'] = ['_'.join(x.split('_')[:3]) for x in arr.index]
    arr['Motif'] = [x.split('_')[3] if len(x.split('_'))>3 else '' for x in arr.index] 
    arr.set_index(['type', 'Motif'], append = True, inplace = True)
    arr = arr.groupby('Motif').sum()
    arr = arr.T
    return arr


def split_train_test(atac, 
                     trainc = None, 
                     testc = None, 
                     prop = 2/3, 
                     split_pref = True,
                     random_seed = 57):
    num_train = int(atac.shape[0]*prop)
    if trainc is None or testc is None:
        np.random.seed(random_seed)
        trainc = atac.obs.index[np.random.choice(atac.shape[0], num_train, replace = False)]
        testc = pd.Index(set(atac.obs.index)-set(trainc))
        split_pref = False
    if split_pref:
        trainc = ['ATAC-'+x for x in trainc]
        testc = ['ATAC-'+x for x in testc]
    train_atac = atac[trainc,:].copy()
    test_atac = atac[testc,:].copy()
    valid_peaks = train_atac.var.index[np.ravel(train_atac.X.sum(0)>0)].intersection(test_atac.var.index[np.ravel(test_atac.X.sum(0)>0)])
    train_atac = train_atac[:, valid_peaks]
    test_atac = test_atac[:, valid_peaks]
    train_atac.layers['tf_idf'] = mv.tfidf_norm(train_atac, copy = True).X
    train_atac.X = np.array(train_atac.layers["poisson_corrected"].copy())
    sc.experimental.pp.normalize_pearson_residuals(train_atac, theta=1)
    train_atac.layers["pearson"] = np.nan_to_num(train_atac.X).copy()
    train_atac.layers['zscored'] = np.nan_to_num(scipy.stats.zscore(train_atac.layers['libnorm']))
    train_atac.layers['zscored_pearson'] = np.nan_to_num(scipy.stats.zscore(train_atac.layers['pearson']))
    test_atac.layers['tf_idf'] = mv.tfidf_norm(test_atac, copy = True).X
    test_atac.X = np.array(test_atac.layers["poisson_corrected"].copy())
    sc.experimental.pp.normalize_pearson_residuals(test_atac, theta=1)
    test_atac.layers["pearson"] = np.nan_to_num(test_atac.X).copy()
    test_atac.layers['zscored'] = np.nan_to_num(scipy.stats.zscore(test_atac.layers['libnorm']))
    test_atac.layers['zscored_pearson'] = np.nan_to_num(scipy.stats.zscore(test_atac.layers['pearson']))
    return train_atac, test_atac