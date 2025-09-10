import numpy as np
import scipy
import pandas as pd
def my_linear_pvals(coefs, X, y, fit_C, inter = True):
    B =coefs
    # because of intercept
    # if inter:
    #     X = np.concatenate([np.ones(X.shape[0]).reshape(-1,1), X], axis = 1)
    XX = pd.DataFrame(X).copy()
    XX = XX.subtract(XX.mean(0),1).values
    yy = (y-np.mean(y)).copy()
    ## lambda may be a vector
    l = fit_C#1/(2*fit_C)
    #print(l)
    n = XX.shape[0]
    U, D, V = scipy.linalg.svd(XX, full_matrices = False)
    V = V.T
    D2 = D**2
    div = D2+l
    def crossprod(m):
        return np.matmul(m.T, m)
    def somefunc():
        # print(U.shape)
        # print(np.diag((D2)/(x[-1,0])).shape)
        # print(y.shape)
        return crossprod(yy - np.matmul(np.matmul(np.matmul(U, np.diag((D2)/(div))),U.T),yy)) / (n - np.sum(D2 * (D2 + 2 * l) / (div**2)))
    sig2hat = somefunc()
    varmat = np.matmul(np.matmul(V,np.diag(D2 / (div**2))), V.T)
    
    varmat = sig2hat * varmat
    se =np.sqrt(np.diag(varmat))
    tstat=np.abs(B / se)
    pval = 2*(1 - scipy.stats.norm.cdf(np.abs(tstat)))
    return pval

