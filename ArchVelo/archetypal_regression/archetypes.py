import numpy as np
import pandas as pd
import scanpy as sc

from py_pcha import PCHA

# archetypal analysis (AA) for ATAC modality without splitting into train and test sets; 
# only specifies in and out directories which should contain data in correct format
def apply_AA_no_test(atac, outdir = '', k = 8, delta = 0.1, conv_crit=1E-6, maxiter = 200):
    # indir: directory containing processed data
    # outdir: directory where the results will be saved
    # k: number of archetypes
    # maxiter: max number of iterations for AA
    
    XC, S = create_archetypes_no_test(atac,
                      num_comps =k, 
                      delta = delta,
                      conv_crit=conv_crit,
                      maxiter = maxiter,
                      outdir = outdir,
                      layer = 'pearson',
                      verbose = True)

    return XC, S

# archetypal analysis (AA) for ATAC modality, splitting into train and test sets; 
# only specifies in and out directories which should contain data in correct format
    
def apply_AA(indir, outdir, k = 8, conv_crit=1E-6, maxiter = 200):
    # indir: directory containing processed data
    # outdir: directory where the results will be saved
    # k: number of archetypes
    # maxiter: max number of iterations for AA
    
    adata_atac_raw = sc.read_h5ad(indir+'processed_data/adata_atac_raw.h5ad')
    atac = adata_atac_raw
    
    np.random.seed(57)
    trainc = np.random.choice(atac.shape[0], int(atac.shape[0]*3/4), replace = False)
    testc = list(set(range(atac.shape[0]))-set(trainc))
    pd.Series(trainc).to_csv(indir+'processed_data/trainc.csv')
    pd.Series(testc).to_csv(indir+'processed_data/testc.csv')
    
    train_atac = atac[trainc,:].copy()
    test_atac = atac[testc,:].copy()
    XC_train, XC_test, S, XC_train_folds, XC_test_folds, S_folds = create_archetypes(train_atac, test_atac, 
                      num_comps =k,
                      conv_crit = conv_crit,
                      maxiter = maxiter,
                      outdir = outdir,
                      layer = 'pearson',
                      create_folds = False,
                      fold_index = None,
                      verbose = True)                                                              
    
# archetypal analysis (AA) for ATAC modality, dataset explicitly supplied

def create_archetypes_no_test(atac,
                      num_comps = 8,
                      delta = 0.1,
                      conv_crit=1E-6,
                      maxiter = 50,
                      outdir = '',
                      layer = 'pearson',
                      verbose = False):
    
    # atac: anndata object containing ATAC modality
    # num_comps: number of components for AA
    # delta: relaxation parameter for delta-AA method (see PCHA)
    # maxiter: max number of iterations for AA
    # outdir: output directory for analysis
    # layer: which layer of atac anndata object to analyze
    # verbose: verbosity parameter for PCHA
    
    to_cluster = atac.layers[layer]
    
    print('Applying AA...')
    res = PCHA(np.array(to_cluster), 
               noc = num_comps, 
               delta = delta,
               conv_crit = conv_crit, 
               maxiter = maxiter, 
               verbose = verbose)
    #save results
    XC, S, C, SSE, varexpl = res
                
    XC = pd.DataFrame(XC, index = atac.obs.index)
    S = pd.DataFrame(S, columns = atac.var.index)
    C = pd.DataFrame(C, index = atac.var.index)

    XC.to_csv(outdir+'/cell_on_peaks_'+str(num_comps)+'_comps.csv')
    S.to_csv(outdir+'/peak_on_peaks_'+str(num_comps)+'_comps.csv')
    C.to_csv(outdir+'/C_train_on_peaks_'+str(num_comps)+'_comps.csv')
    
    
    return XC, S
    
# archetypal analysis (AA) for ATAC modality, train and test sets explicitly supplied

def create_archetypes(train_atac, test_atac, 
                      num_comps = 8,
                      delta = 0.1,
                      conv_crit=1E-6,
                      maxiter = 50,
                      outdir = '',
                      layer = 'pearson',
                      transposed = False,
                      create_folds = False,
                      only_folds = False,
                      fold_index = None,
                      verbose = False):
    # train_atac: anndata object containing ATAC modality, train set
    # test_atac: anndata object containing ATAC modality, test set
    # num_comps: number of components for AA
    # delta: relaxation parameter for delta-AA method (see PCHA)
    # maxiter: max number of iterations for AA
    # outdir: output directory for analysis
    # layer: which layer of atac anndata object to analyze
    # create_folds: create folds for cross-validation?
    # only_folds: only write out data for cross-validation?
    # fold_index: index vector coding the data split for cross-validation
    # verbose: verbosity parameter for PCHA
    
    to_cluster = train_atac.layers[layer]
    to_cluster_test = test_atac.layers[layer]
    if not only_folds:
        print('Fitting train data...')
        res = PCHA(np.array(to_cluster), noc = num_comps, delta = delta, conv_crit = conv_crit, maxiter = maxiter, verbose = verbose)
        #save results
        XC, S, C, SSE, varexpl = res
                    
        XC_train = pd.DataFrame(XC, index = train_atac.obs.index)
        XC_test = np.matmul(np.array(to_cluster_test), C)
        S = pd.DataFrame(S, columns = train_atac.var.index)
        C = pd.DataFrame(C, index = train_atac.var.index)
        XC_test = pd.DataFrame(XC_test, index = test_atac.obs.index)

        XC_train.to_csv(outdir+'/cell_train_on_peaks_'+str(num_comps)+'_comps.csv')
        XC_test.to_csv(outdir+'/cell_test_on_peaks_'+str(num_comps)+'_comps.csv')
        S.to_csv(outdir+'/peak_on_peaks_'+str(num_comps)+'_comps.csv')
        C.to_csv(outdir+'/C_train_on_peaks_'+str(num_comps)+'_comps.csv')
    
    XC_train_folds = {}
    XC_test_folds = {}
    S_folds = {}
    if create_folds:
        l = fold_index
        all_cluster = to_cluster
        for fld in np.unique(fold_index):
            print('Fold '+str(fld))
            to_cluster_train = all_cluster[l != fld,:]
            to_cluster_test = all_cluster[l == fld,:]
            #if fld>0:
            print('Fitting train data...')
            res = PCHA(to_cluster_train, noc = num_comps, delta = delta, conv_crit = conv_crit, maxiter = maxiter, verbose = verbose)
            XC, S, C, SSE, varexpl = res
            
            XC_test = np.matmul(np.array(to_cluster_test), C)
            
            S = pd.DataFrame(S, columns = train_atac.var.index)
            XC_train = pd.DataFrame(XC, index = train_atac.obs.index[l!=fld])
            print('Transforming test data...')
                        
            XC_test = pd.DataFrame(XC_test, index = train_atac.obs.index[l==fld])
            
            XC_train_folds[fld] = XC_train
            XC_test_folds[fld] = XC_test
            S_folds[fld] = S
            XC_train.to_csv(outdir+'/cell_train_on_peaks_'+str(num_comps)+'_comps_fold_'+str(fld)+'.csv')
            XC_test.to_csv(outdir+'/cell_test_on_peaks_'+str(num_comps)+'_comps_fold_'+str(fld)+'.csv')
            S.to_csv(outdir+'/peak_on_peaks_'+str(num_comps)+'_comps_fold_'+str(fld)+'.csv')
    
    return XC_train, XC_test, S, XC_train_folds, XC_test_folds, S_folds
