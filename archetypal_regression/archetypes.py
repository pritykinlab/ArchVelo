import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import anndata
import scanpy as sc
import pickle

from py_pcha import PCHA
from py_pcha.furthest_sum import furthest_sum
import archetypes as arch
from sklearn.utils.validation import check_is_fitted, check_random_state

from datetime import datetime as dt
import time

def apply_AA_no_test(indir, outdir, k = 8, maxiter = 200):
    adata_atac_raw = sc.read_h5ad(indir+'processed_data/adata_atac_raw.h5ad')
    atac = adata_atac_raw
    XC, S = create_archetypes_no_test(atac,
                      num_comps =k,maxiter = maxiter,
                      outdir = outdir,
                      layer = 'pearson',
                      verbose = True)

#def create_train_test_split()
    
def apply_AA(indir, outdir, k = 8, maxiter = 200):
    adata_atac_raw = sc.read_h5ad(indir+'processed_data/adata_atac_raw.h5ad')
    
    np.random.seed(57)
    trainc = np.random.choice(atac.shape[0], int(atac.shape[0]*3/4), replace = False)
    testc = list(set(range(atac.shape[0]))-set(trainc))
    pd.Series(trainc).to_csv(indir+'processed_data/trainc.csv')
    pd.Series(testc).to_csv(indir+'processed_data/testc.csv')
    
    train_atac = atac[trainc,:].copy()
    test_atac = atac[testc,:].copy()
    XC_train, XC_test, S, XC_train_folds, XC_test_folds, S_folds = create_archetypes(train_atac, test_atac, 
                      num_comps =k,maxiter = maxiter,
                      outdir = outdir,
                      layer = 'pearson',
                      create_folds = False,
                      fold_index = None,
                      verbose = True)

def create_archetypes_no_test(atac,
                      num_comps,
                      delta = 0.1,
                      maxiter = 50,
                      outdir = '',
                      layer = 'pearson',
                      verbose = False):
    
    to_cluster = atac.layers[layer]
    
    print('Applying AA...')
    res = PCHA(np.array(to_cluster), noc = num_comps, delta = delta, maxiter = maxiter, verbose = verbose)
    #save results
    XC, S, C, SSE, varexpl = res
                
    XC = pd.DataFrame(XC, index = atac.obs.index)
    S = pd.DataFrame(S, columns = atac.var.index)
    C = pd.DataFrame(C, index = atac.var.index)

    XC.to_csv(outdir+'/cell_on_peaks_'+str(num_comps)+'_comps.csv')
    S.to_csv(outdir+'/peak_on_peaks_'+str(num_comps)+'_comps.csv')
    C.to_csv(outdir+'/C_train_on_peaks_'+str(num_comps)+'_comps.csv')
    
    
    return XC, S


def create_archetypes(train_atac, test_atac, 
                      num_comps,
                      delta = 0.1,
                      maxiter = 50,
                      outdir = '',
                      layer = 'pearson',
                      transposed = False,
                      create_folds = False,
                      only_folds = False,
                      fold_index = None,
                      verbose = False):
    
    to_cluster = train_atac.layers[layer]
    to_cluster_test = test_atac.layers[layer]
    if not only_folds:
        print('Fitting train data...')
        res = PCHA(np.array(to_cluster), noc = num_comps, delta = delta, maxiter = maxiter, verbose = verbose)
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
            res = PCHA(to_cluster_train, noc = num_comps, delta = delta, maxiter = maxiter, verbose = verbose)
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


# from archetypes.algorithms.archetypes import _optimize_betas
# def aa_simple(X, i_alphas, i_betas, max_iter, tol, verbose=False):
#     alphas = i_alphas
#     betas = i_betas

#     Z = betas @ X

#     rss_0 = np.inf
#     for n_iter in range(max_iter):
#         print(n_iter)
#         if verbose and n_iter % 100 == 0:
#             print(f"    Iteration: {n_iter + 1:{len(str(max_iter))}}, RSS: {rss_0:.2f}")

#         #alphas = _optimize_alphas(X, Z)
#         Z = np.linalg.pinv(alphas) @ X
#         betas = _optimize_betas(Z, X)
#         Z = betas @ X
#         rss = np.linalg.norm(X - alphas @ Z)  # Frobenius norm
#         if np.abs(rss_0 - rss) < tol:
#             break
#         rss_0 = rss

#     return alphas, betas, rss_0, Z, n_iter

# def transform_test(n_arch, X, S, maxiter = 50, verbose = False):
#     aa1 = arch.AA(n_archetypes=n_arch, algorithm_init="auto")
#     aa1._algorithm_init = "auto"
#     aa1.rss_ = np.inf
#     aa1.verbose = verbose
#     aa1.max_iter = maxiter
#     #X = to_cluster_test.values.T
#     for i in range(aa1.n_init):
#         print(i)
#         if aa1.verbose:
#             print(f"Initialization {i + 1:{len(str(aa1.n_init))}}/{aa1.n_init}")

#         i_alphas, i_betas = aa1._init_coefs(X, random_state = check_random_state(None))
#         i_alphas = S.T

#         alphas, betas, rss, Z, n_iter = aa_simple(
#             X, i_alphas, i_betas, aa1.max_iter, aa1.tol, aa1.verbose
#         )

#         if rss < aa1.rss_:
#             aa1.alphas_ = alphas
#             aa1.betas_ = betas
#             aa1.archetypes_ = Z
#             aa1.n_iter_ = n_iter
#             aa1.rss_ = rss
#     return Z.T

# def create_archetypes_old(train_atac, test_atac, 
#                       num_comps,
#                       delta = 0.1,
#                       maxiter = 50,
#                       outdir = '',
#                       flag = 'new',
#                       layer = 'X_magic',
#                       transposed = False,
#                       create_folds = False,
#                       only_folds = False,
#                       fold_index = None,
#                       verbose = False):
    
#     to_cluster = train_atac.layers[layer]
#     to_cluster_test = test_atac.layers[layer]
#     if not only_folds:
#         print('Fitting train data...')
#         res = PCHA(np.array(to_cluster), noc = num_comps, delta = delta, maxiter = maxiter, verbose = verbose)
#         #save results
#         XC, S, C, SSE, varexpl = res
#         # S = S.T
#         # S['gene'] = peaks_to_genes.loc[S.index]['gene']
#         # S.set_index('gene', append = True, inplace = True)
#         # S = S.T
#         print('Transforming test data...')
#         XC_test = transform_test(num_comps, np.array(to_cluster_test).T, S, maxiter = maxiter, verbose = verbose)
#         XC_test = pd.DataFrame(XC_test, index = test_atac.obs.index)

#         if not flag:
#             XC_test, _, _, _, _ = res = PCHA_transform(np.array(to_cluster_test), noc = num_comps, 
#                                  S_fixed = S, delta = delta, maxiter = maxiter, verbose = verbose)

#         else:
#             XC_test_alt = np.matmul(np.array(to_cluster_test), C)
#             XC_test = XC_test_alt
                    
#         XC_train = pd.DataFrame(XC, index = train_atac.obs.index)
#         S = pd.DataFrame(S, columns = train_atac.var.index)
#         C = pd.DataFrame(C, index = train_atac.var.index)
#         if not flag:
#             XC_test = pd.DataFrame(XC_test, index = test_atac.obs.index)
#         XC_test_alt = pd.DataFrame(XC_test_alt, index = test_atac.obs.index)

#         XC_train.to_csv(outdir+'/cell_train_on_peaks_'+str(num_comps)+'_comps.csv')
#         if not flag:
#             XC_test.to_csv(outdir+'/cell_test_on_peaks_'+str(num_comps)+'_comps.csv')
#         XC_test_alt.to_csv(outdir+'/alt_cell_test_on_peaks_'+str(num_comps)+'_comps.csv')
#         S.to_csv(outdir+'/peak_on_peaks_'+str(num_comps)+'_comps.csv')
#         C.to_csv(outdir+'/C_train_on_peaks_'+str(num_comps)+'_comps.csv')
    
#     XC_train_folds = {}
#     XC_test_folds = {}
#     S_folds = {}
#     if create_folds:
#         l = fold_index
#         all_cluster = to_cluster
#         for fld in np.unique(fold_index):
#             print('Fold '+str(fld))
#             to_cluster_train = all_cluster[l != fld,:]
#             to_cluster_test = all_cluster[l == fld,:]
#             #if fld>0:
#             print('Fitting train data...')
#             res = PCHA(to_cluster_train, noc = num_comps, delta = delta, maxiter = maxiter, verbose = verbose)
#             XC, S, C, SSE, varexpl = res
            
#             XC_test, _, _, _, _ = PCHA_transform(np.array(to_cluster_test), noc = num_comps, 
#                              S_fixed = S, delta = delta, maxiter = maxiter, verbose = verbose)
            
#             S = pd.DataFrame(S, columns = train_atac.var.index)
#             XC_train = pd.DataFrame(XC, index = train_atac.obs.index[l!=fld])
#             print('Transforming test data...')
                        
#             XC_test = pd.DataFrame(XC_test, index = train_atac.obs.index[l==fld])
            
#             XC_train_folds[fld] = XC_train
#             XC_test_folds[fld] = XC_test
#             S_folds[fld] = S
#             XC_train.to_csv(outdir+'/cell_train_on_peaks_'+str(num_comps)+'_comps_fold_'+str(fld)+'.csv')
#             XC_test.to_csv(outdir+'/cell_test_on_peaks_'+str(num_comps)+'_comps_fold_'+str(fld)+'.csv')
#             S.to_csv(outdir+'/peak_on_peaks_'+str(num_comps)+'_comps_fold_'+str(fld)+'.csv')
    
#     return XC_train, XC_test, S, XC_train_folds, XC_test_folds, S_folds

 

# """Principal Convex Hull Analysis (PCHA) / Archetypal Analysis."""

# def PCHA_transform(X, noc, S_fixed = None, I=None, U=None, delta=0, verbose=False, conv_crit=1E-6, maxiter=500):
#     """Return archetypes of dataset.
#     Note: Commonly data is formatted to have shape (examples, dimensions).
#     This function takes input and returns output of the transposed shape,
#     (dimensions, examples).
#     Parameters
#     ----------
#     X : numpy.2darray
#         Data matrix in which to find archetypes
#     noc : int
#         Number of archetypes to find
#     I : 1d-array
#         Entries of X to use for dictionary in C (optional)
#     U : 1d-array
#         Entries of X to model in S (optional)
#     Output
#     ------
#     XC : numpy.2darray
#         I x noc feature matrix (i.e. XC=X[:,I]*C forming the archetypes)
#     S : numpy.2darray
#         noc x length(U) matrix, S>=0 |S_j|_1=1
#     C : numpy.2darray
#         noc x length(U) matrix, S>=0 |S_j|_1=1
#     SSE : float
#         Sum of Squared Errors
#     varexlp : float
#         Percent variation explained by the model
#     """

#     def C_update(X, XSt, XC, SSt, C, delta, muC, mualpha, SST, SSE, niter=1):
#         """Update C for one iteration of the algorithm."""
#         J, nos = C.shape

#         if delta != 0:
#             alphaC = np.sum(C, axis=0).A[0]
#             C = np.dot(C, np.diag(1 / alphaC))

#         e = np.ones((J, 1))
#         XtXSt = np.dot(X.T, XSt)

#         for k in range(niter):

#             # Update C
#             SSE_old = SSE
#             g = (np.dot(X.T, np.dot(XC, SSt)) - XtXSt) / SST

#             if delta != 0:
#                 g = np.dot(g, np.diag(alphaC))
#             g = g.A - e * np.sum(g.A * C.A, axis=0)

#             C_old = C
#             while True:
#                 C = (C_old - muC * g).clip(min=0)
#                 nC = np.sum(C, axis=0) + np.finfo(float).eps
#                 C = np.dot(C, np.diag(1 / nC.A[0]))

#                 if delta != 0:
#                     Ct = C * np.diag(alphaC)
#                 else:
#                     Ct = C

#                 XC = np.dot(X, Ct)
#                 CtXtXC = np.dot(XC.T, XC)
#                 SSE = SST - 2 * np.sum(XC.A * XSt.A) + np.sum(CtXtXC.A * SSt.A)
                

#                 if SSE <= SSE_old * (1 + 1e-9):
#                     muC = muC * 1.2
#                     break
#                 else:
#                     muC = muC / 2

#             # Update alphaC
#             SSE_old = SSE
#             if delta != 0:
#                 g = (np.diag(CtXtXC * SSt).T / alphaC - np.sum(C.A * XtXSt.A)) / (SST * J)
#                 alphaC_old = alphaC
#                 while True:
#                     alphaC = alphaC_old - mualpha * g
#                     alphaC[alphaC < 1 - delta] = 1 - delta
#                     alphaC[alphaC > 1 + delta] = 1 + delta

#                     XCt = np.dot(XC, np.diag(alphaC / alphaC_old))
#                     CtXtXC = np.dot(XCt.T, XCt)
#                     SSE = SST - 2 * np.sum(XCt.A * XSt.A) + np.sum(CtXtXC.A * SSt.A)

#                     if SSE <= SSE_old * (1 + 1e-9):
#                         mualpha = mualpha * 1.2
#                         XC = XCt
#                         break
#                     else:
#                         mualpha = mualpha / 2

#         if delta != 0:
#             C = C * np.diag(alphaC)

#         return C, SSE, muC, mualpha, CtXtXC, XC

#     N, M = X.shape
    

#     if I is None:
#         I = range(M)
#     if U is None:
#         U = range(M)

#     SST = np.sum(X[:, U] * X[:, U])

#     # Initialize C
#     try:
#         i = furthest_sum(X[:, I], noc, [int(np.ceil(len(I) * np.random.rand()))])
#     except IndexError:
#         class InitializationException(Exception): pass
#         raise InitializationException("Initialization does not converge. Too few examples in dataset.")

#     j = range(noc)
#     C = csr_matrix((np.ones(len(i)), (i, j)), shape=(len(I), noc)).todense()

#     XC = np.dot(X[:, I], C)

#     muS, muC, mualpha = 1, 1, 1

#     # Initialise S
#     XCtX = np.dot(XC.T, X[:, U])
#     CtXtXC = np.dot(XC.T, XC)
#     # S = -np.log(np.random.random((noc, len(U))))
#     # S = S / np.dot(np.ones((noc, 1)), np.mat(np.sum(S, axis=0)))
#     S = S_fixed
#     SSt = np.dot(S, S.T)
#     SSE = SST - 2 * np.sum(XCtX.A * S.A) + np.sum(CtXtXC.A * SSt.A)
#     #S, SSE, muS, SSt = S_update(S, XCtX, CtXtXC, muS, SST, SSE, 25)

#     # Set PCHA parameters
#     iter_ = 0
#     dSSE = np.inf
#     t1 = dt.now()
#     varexpl = (SST - SSE) / SST

#     if verbose:
#         print('\nPrincipal Convex Hull Analysis / Archetypal Analysis')
#         print('A ' + str(noc) + ' component model will be fitted')
#         print('To stop algorithm press control C\n')

#     dheader = '%10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s' % ('Iteration', 'Expl. var.', 'Cost func.', 'Delta SSEf.', 'muC', 'mualpha', 'muS', ' Time(s)   ')
#     dline = '-----------+------------+------------+-------------+------------+------------+------------+------------+'

#     while np.abs(dSSE) >= conv_crit * np.abs(SSE) and iter_ < maxiter and varexpl < 0.9999:
#         if verbose and iter_ % 100 == 0:
#             print(dline)
#             print(dheader)
#             print(dline)
#         told = t1
#         iter_ += 1
#         SSE_old = SSE

#         # C (and alpha) update
#         XSt = np.dot(X[:, U], S.T)
#         C, SSE, muC, mualpha, CtXtXC, XC = C_update(
#             X[:, I], XSt, XC, SSt, C, delta, muC, mualpha, SST, SSE, 10
#         )

#         # S update
#         XCtX = np.dot(XC.T, X[:, U])
#         # S, SSE, muS, SSt = S_update(
#         #     S, XCtX, CtXtXC, muS, SST, SSE, 10
#         # )

#         # Evaluate and display iteration
#         dSSE = SSE_old - SSE
#         t1 = dt.now()
#         if iter_ % 1 == 0:
#             time.sleep(0.000001)
#             varexpl = (SST - SSE) / SST
#             if verbose:
#                 print('%10.0f | %10.4f | %10.4e | %10.4e | %10.4e | %10.4e | %10.4e | %10.4f \n' % (iter_, varexpl, SSE, dSSE/np.abs(SSE), muC, mualpha, muS, (t1-told).seconds))

#     # Display final iteration
#     varexpl = (SST - SSE) / SST
#     if verbose:
#         print(dline)
#         print(dline)
#         print('%10.0f | %10.4f | %10.4e | %10.4e | %10.4e | %10.4e | %10.4e | %10.4f \n' % (iter_, varexpl, SSE, dSSE/np.abs(SSE), muC, mualpha, muS, (t1-told).seconds))

#     # # Sort components according to importance
#     # ind, vals = zip(
#     #     *sorted(enumerate(np.sum(S, axis=1)), key=lambda x: x[0], reverse=1)
#     # )
#     # S = S[ind, :]
#     # C = C[:, ind]
#     # XC = XC[:, ind]

#     return XC, S, C, SSE, varexpl

        
    

    