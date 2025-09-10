import numpy as np
import scipy
import pandas as pd
import anndata
import scanpy as sc
from sklearn.linear_model import Lasso, Ridge
import pickle

import sys
sys.path.append('/mnt/home/mavdeeva/ceph/scvelo/Yura_project/atac_regression_package/')
from util import *
from util_atac import *
from plotting import *
from util_regression import *


    
def generate_features(g,
                     peaks_to_genes,
                     train_df,
                     test_df,        
                     cell_train_comps,
                     cell_test_comps,
                     peak_comps,
                     motif_mat,
                     atac_layer = 'X_magic',
                     feature_type = 'soft_clust'):  
    
    rel_peaks = peaks_to_genes.query('gene == "'+str(g)+'"').index
    if motif_mat is not None:
        unique_peaks = np.unique(motif_mat.index)
        rel_peaks = rel_peaks.intersection(pd.Index(unique_peaks))
    #print(rel_peaks)
    
    rel_peaks = rel_peaks.intersection(train_df.var.index).intersection(test_df.var.index)
    rel_peaks_index = rel_peaks

    if feature_type!='soft_clust':
        cell_by_mot_train = pd.DataFrame(np.array(train_df[:, rel_peaks].layers[atac_layer]).astype(float), 
                                         index = train_df.obs.index,
                                         columns = rel_peaks_index)
        #print(cell_by_mot_train.iloc[:3, :3])
        cell_by_mot_test = pd.DataFrame(np.array(test_df[:, rel_peaks].layers[atac_layer]).astype(float),
                                        index = test_df.obs.index,
                                        columns = rel_peaks_index)
    else:
        rel_cells_train = train_df.obs.index
        rel_cells_test = test_df.obs.index

        pre_train = cell_train_comps.loc[rel_cells_train,:].astype(float)
        pre_test = cell_test_comps.loc[rel_cells_test,:].astype(float)
        num_comps = cell_train_comps.shape[1]
        rel_peak_weights = peak_comps.loc[:, rel_peaks]
        cell_by_mot_train = {}
        cell_by_mot_test = {}
        types = []
        for i in range(num_comps):
            cell_by_mot_train['Peak_clust_'+str(i)] = pd.DataFrame(np.matmul(pre_train.iloc[:,[i]].values, rel_peak_weights.iloc[[i],:].values),
             index = pre_train.index, columns = rel_peak_weights.columns)
            cell_by_mot_test['Peak_clust_'+str(i)] = pd.DataFrame(np.matmul(pre_test.iloc[:,[i]].values, rel_peak_weights.iloc[[i],:].values),
             index = pre_test.index, columns = rel_peak_weights.columns)
            types.append('Peak_clust_'+str(i))
        cell_by_mot_train = pd.concat(cell_by_mot_train, axis = 1)
        cell_by_mot_train.columns.names = ['type', 'peak']
        cell_by_mot_test = pd.concat(cell_by_mot_test, axis = 1)
        cell_by_mot_test.columns.names = ['type', 'peak']
        rel_peaks_index = cell_by_mot_train.columns
    if motif_mat is not None: 
        if feature_type!='soft_clust':
            rel_mots = motif_mat.loc[rel_peaks,:].astype(float)

            cell_by_mot_train = cell_by_mot_train.dot(rel_mots)
            cell_by_mot_test = cell_by_mot_test.dot(rel_mots)
        else:
            by_type_train = []
            by_type_test = []
            by_mots = []
            for typ in np.unique(types):
                #print(cell_by_mot_train)
                rel_train = cell_by_mot_train.loc[:,typ]
                rel_test = cell_by_mot_test.loc[:,typ]
                ann_rel_peaks = rel_train.columns
                ann_rel_mots = motif_mat.loc[ann_rel_peaks,:].astype(float)
                ann_rel_mots.columns = [typ+'_'+x for x in ann_rel_mots.columns]
                by_type_train.append(rel_train.dot(ann_rel_mots))
                by_type_test.append(rel_test.dot(ann_rel_mots))
                by_mots.append(ann_rel_mots)
            cell_by_mot_train = pd.concat(by_type_train, axis = 1)
            cell_by_mot_test = pd.concat(by_type_test, axis = 1)
            rel_mots = pd.concat(by_mots, axis = 1)
        to_leave = (rel_mots.sum(0)>0)
    else:
        if feature_type!='soft_clust':
            to_leave = [True]*len(rel_peaks)
        else:
            to_leave = [True]*cell_train_comps.shape[1]
            cell_by_mot_train = cell_by_mot_train.T.groupby('type').sum().T
            #print(cell_by_mot_train)
            #print(to_leave)
            cell_by_mot_test = cell_by_mot_test.T.groupby('type').sum().T
    #print(rel_peaks)
    cell_by_mot_train = cell_by_mot_train.loc[:, to_leave]
    cell_by_mot_test = cell_by_mot_test.loc[:, to_leave]
    return cell_by_mot_train, cell_by_mot_test
    
def rna_target(g, rna_data,
               rna_layer = 'Ms',
               verbose = False):
    if verbose:
        print('RNA target, from layer '+rna_layer)
    y = rna_data[:,g].layers[rna_layer]
    return y

def velocity_target(g, path_to_velocity_models,
                    rna_data = None,
                    dep_var = '3',
                    sort_ids = None,
                    return_times = False,
                    time_mode = 'orig',
                    lik_thres = 0,
                    remove_steady = False,
                    normalize_probs = True,
                    verbose = False):
                    
    out = path_to_velocity_models
    f = open(out+g+'.p', 'rb')
    dm = pickle.load(f)
    f.close()
    
    if verbose:
        print('Velocity target, state: '+dep_var)
        
    # generate target variable
    if dep_var in ['0', '1', '2', '3']:
        i = int(dep_var)
        y = dm.get_divergence(mode = "likelihood", normalized = normalize_probs)[i,:]
    elif dep_var == 'phase':
        #print('phase')
        phase = dm.get_divergence(mode = "hard_state")
        y = 2*phase-1
    elif dep_var == 'all':
        indiv_s = {}
        for i in range(4):
            indiv_s[i] = dm.get_divergence(mode = "likelihood", normalized = normalize_probs)[i,:]
        y = np.concatenate([indiv_s[i].reshape(-1,1) for i in range(4)], 1)
    
    if dep_var !='all':
        y = y.reshape(-1,1)
    y = y[sort_ids,:]
    
    # remove some cells
    if lik_thres>0:
        ss = dm.get_divergence(mode = "soft_state", normalized = normalize_probs)[sort_ids]
        ss_quant = (np.abs(ss)>=lik_thres)
    else:
        ss_quant = [True]*len(sort_ids)
    if remove_steady:    
        phase = dm.get_divergence(mode = "hard_state")[sort_ids]
        ss_quant*=(phase <2)
    if verbose:
        print(str(-np.sum(ss_quant)+len(sort_ids))+' cells to be removed')
            
    if return_times:
        if time_mode == 'orig' or rna_data is None:
            if verbose:
                print('Returning times without smoothing.')
            time = dm.t[sort_ids]
        else:
            if verbose:
                print('Returning smoothed times.')
            time = np.ravel(rna_data[:,g].layers['fit_t'])
    else:
        time = None
    return y, ss_quant, time
                    
def generate_train_test_data(g,
                      rna_data, 
                      train_df, 
                      test_df,
                      dep_var = '3',
                      rna_layer = 'Ms', 
                      atac_layer = 'X_magic',
                      sort_ids = None,
                      peaks_to_genes = None,
                      feature_type = 'soft_clust',                      
                      path_to_velocity_models = None,
                      return_times = False,
                      time_mode = 'orig',
                      cell_train_comps = None,
                      cell_test_comps = None,
                      peak_comps = None,
                      smooth_y = False,
                      remove_steady = True,
                      inverse_sigmoid = True,
                      lg = False,
                      train_ind_orig = None, 
                      test_ind_orig = None,
                      motif_mat = None,                  
                      normalize_probs = True,
                      lik_thres = 0.0,
                      verbose = False):
    
    cell_by_mot_train, cell_by_mot_test = generate_features(g,                                                                        peaks_to_genes,
                     train_df,
                     test_df,
                     cell_train_comps,
                     cell_test_comps,
                     peak_comps,
                     motif_mat,
                     atac_layer = atac_layer,
                     feature_type = feature_type)
    
    X_train = cell_by_mot_train
    X_test = cell_by_mot_test
    
    if dep_var == 'rna':
        y = np.array(rna_target(g, rna_data, rna_layer = rna_layer,
               verbose = verbose))
    else:
        if sort_ids is None:
            sort_ids = range(rna_data.shape[0])
        y, ss_quant, time = velocity_target(g, path_to_velocity_models,
                    rna_data = None,
                    dep_var = dep_var,
                    sort_ids = sort_ids,
                    return_times = return_times,
                    time_mode = 'orig',
                    lik_thres = lik_thres,
                    remove_steady = remove_steady,
                    normalize_probs = normalize_probs,
                    verbose = verbose)
        
        train_allowed = []
        for (i,x) in enumerate(train_ind_orig):
            if x in np.where(ss_quant)[0]:
                train_allowed.append(i)
        test_allowed = []
        for (i,x) in enumerate(test_ind_orig):
            if x in np.where(ss_quant)[0]:
                test_allowed.append(i)
    if inverse_sigmoid and dep_var !='phase':
        y = inverse_sigm(y)
    else:
        if lg and dep_var !='phase':
            y = np.log(y)#-np.log(1-y)
            
    y_train = y[train_ind_orig,:]
    y_test= y[test_ind_orig,:] 
    #print(y_train)
    #print(train_allowed)
    if dep_var != 'rna':
        y_train = y_train[train_allowed,:]
        y_test= y_test[test_allowed,:]
       
        X_train = X_train.iloc[train_allowed,:]
        X_test = X_test.iloc[test_allowed,:]
        
        if return_times:
            time_train = time[train_ind_orig][train_allowed]
            time_test = time[test_ind_orig][test_allowed]
        else:
            time_train = None
            time_test = None
    
        if smooth_y:
            if return_times:
                aa = np.argsort(time_train)
                aaa_train = np.argsort(aa)
                y_train = y_train[aa]
                aa = np.argsort(time_test)
                aaa_test = np.argsort(aa)
                y_test = y_test[aa]
                len_train = int(y_train.shape[0]/30)
                y_train = pd.Series(np.ravel(y_train)).rolling(len_train, center = True, min_periods = 0).mean().values
                len_test = int(y_test.shape[0]/30)
                y_test = pd.Series(np.ravel(y_test)).rolling(len_test, center = True, min_periods = 0).mean().values
                y_train = y_train[aaa_train].reshape(-1,1)
                y_test = y_test[aaa_test].reshape(-1,1)
            else:
                print('Cannot smooth target, please return_times = True')
    else:
        time_train = None
        time_test = None
    #print(y_train)
    return X_train, y_train, X_test, y_test, time_train, time_test

def normalize_features(X_train, X_test,stand_feat = 'z-score'):
    if stand_feat == 'z-score':
            X_train = standardize(X_train).fillna(0)
            X_test = standardize(X_test).fillna(0)
    elif stand_feat == 'minmax':
            X_train = minmax(X_train)
            X_test = minmax(X_test)  
    return X_train, X_test
    
def atac_regression_with_components(
                       genes = None, 
                       dep_var = '3', 
                       normalize_probs = True, 
                       XC_train = None, 
                       XC_test = None,
                       S = None, 
                       cross_validate = True,
                       XC_train_folds = None, 
                       XC_test_folds = None,
                       S_folds = None, 
                       rna_data = None,
                       train_df = None, 
                       test_df = None, 
                       train_ind_orig = None, 
                       test_ind_orig = None,
                       sort_ids = None,
                       positive = False,
                       reg_type = 'ridge',
                       plot_cv = False,
                       alphas = np.logspace(-7, 5, 50),
                       alpha_to_use = 0.1, 
                       motif_mat = None, 
                       fit_intercept = True,
                       inverse_sigmoid = True,
                       lg = False,
                       peaks_to_genes = None,
                       feature_type = 'soft_clust',
                       #split_types = True,
                       #collapse_types = False,
                       path_to_velocity_models = None,
                       remove_steady = False,
                       lik_thres = 0.0,            
                       penal = 'l2', 
                       metr = 'r2',
                       stand_feat = False, 
                       thres = 0., 
                       rna_layer = 'Ms',
                       atac_layer = 'X_magic',
                       plot = False,
                       verbose = False):
    
    all_coeffs= {}
    valid_scores = {}
    test_scores = {}
    base_scores = {}
    diff_scores = {}
    pvals = {}
    y_train_preds = {}
    y_test_preds = {}
    
    if cross_validate:
        #setup folds
        num_cells = train_df.shape[0]+test_df.shape[0]
        l = [0]*int(num_cells/4)+[1]*int(num_cells/4)+[2]*(len(train_ind_orig) - 2*int(num_cells/4))
        arr_l = np.array(l)
        
    for g in genes:
        if verbose:
            print(g)
        try:
            scrs_valid = {}
            if cross_validate:
                scrs_train = {}
                for fld in [0,1,2]:
                    XC_cur = XC_train_folds[fld]
                    XC_test_cur = XC_test_folds[fld]
                    S_cur = S_folds[fld]
                    #S_cur = S_cur.T
                    X_train, y_train, X_test, y_test, _, _, = generate_train_test_data(g = g,
                            dep_var = dep_var,                                                           
                            rna_data = rna_data,
                            rna_layer = rna_layer,
                            atac_layer = atac_layer,
                            sort_ids = sort_ids,                                                                     
                            path_to_velocity_models = path_to_velocity_models,  
                            feature_type = feature_type,
                            peaks_to_genes = peaks_to_genes,
                            normalize_probs = normalize_probs,
                            cell_train_comps = XC_cur,
                            cell_test_comps = XC_test_cur,
                            peak_comps = S_cur,
                            inverse_sigmoid = inverse_sigmoid,
                            lg = lg,
                            train_df = train_df[arr_l!=fld,:], 
                            test_df = train_df[arr_l==fld,:], 
                            remove_steady = remove_steady,
                            train_ind_orig = list(np.array(train_ind_orig)[arr_l!=fld]), 
                            test_ind_orig = list(np.array(train_ind_orig)[arr_l==fld]),
                            motif_mat = motif_mat, 
                            lik_thres = lik_thres,
                            verbose = verbose)
                    if stand_feat!=False:
                        X_train, X_test = normalize_features(X_train, X_test, stand_feat = stand_feat)
                    scrs_train[fld]={}
                    scrs_valid[fld] = {}
                    for alpha in alphas:
                        if reg_type == 'lasso':               
                            clf = Lasso(fit_intercept = fit_intercept,
                                          alpha = alpha).fit(X_train.values, y_train)
                        else:
                            clf = Ridge(fit_intercept = fit_intercept,
                                          alpha = alpha).fit(X_train.values, y_train)
                        if metr == 'r2':
                            scrs_train[fld][alpha]=clf.score(X_train.values, y_train)
                            scrs_valid[fld][alpha]=clf.score(X_test.values, y_test)
                        elif metr == 'pearson':
                            y_train_pred = clf.predict(X_train.values)
                            y_test_pred = clf.predict(X_test.values)
                            scrs_train[fld][alpha]=scipy.stats.pearsonr(np.ravel(y_train), np.ravel(y_train_pred))[0]
                            scrs_valid[fld][alpha]=scipy.stats.pearsonr(np.ravel(y_test), np.ravel(y_test_pred))[0]
                scrs_train = pd.DataFrame(scrs_train)
                #print(scrs_valid)
                scrs_valid = pd.DataFrame(scrs_valid)
                #print(scrs_valid.shape)
                mean_scrs_valid = scrs_valid.mean(1)
                #print(scrs_valid)
                alpha_best = mean_scrs_valid.index.values[mean_scrs_valid.argmax()]
                if verbose:
                    print('Alpha best: ', alpha_best)
                if plot_cv:
                    plt.figure()
                    ax = plt.gca()
                    plot_with_error(x = np.log10(alphas), 
                                    y = scrs_train.values, 
                                    ax = ax, c = 'green', label = 'train')
                    plot_with_error(x = np.log10(alphas), 
                                    y = scrs_valid.values, 
                                    ax = ax, c = 'blue',label = 'valid')
                    plt.title(g)
                    plt.legend()
            else:
                alpha_best = alpha_to_use
            X_train, y_train, X_test, y_test, _, _ = generate_train_test_data(g = g,
                            dep_var = dep_var,
                            rna_data = rna_data,
                            rna_layer = rna_layer,
                            atac_layer = atac_layer,
                            sort_ids = sort_ids,                                                                     
                            path_to_velocity_models = path_to_velocity_models,
                            feature_type = feature_type,
                            peaks_to_genes = peaks_to_genes,
                            normalize_probs = normalize_probs,
                            cell_train_comps = XC_train,
                            cell_test_comps = XC_test,
                            peak_comps = S,
                            inverse_sigmoid = inverse_sigmoid,
                            lg = lg,
                            train_df = train_df, 
                            test_df = test_df, 
                            remove_steady = remove_steady,
                            train_ind_orig = train_ind_orig, 
                            test_ind_orig = test_ind_orig,
                            motif_mat = motif_mat, 
                            lik_thres = lik_thres,
                            verbose = verbose)
            if verbose:
                print(X_train.shape, y_train.shape)
            if stand_feat!=False:
                X_train, X_test = normalize_features(X_train, X_test, stand_feat = stand_feat)
            #print(X_train.iloc[:3, :3])
            #try:
            #if skl:
            if reg_type == 'lasso':
                clf = Lasso(fit_intercept = fit_intercept,positive = positive,
                            alpha = alpha_best).fit(X_train.values, y_train)
    #             
            else:
                clf = Ridge(fit_intercept = fit_intercept,positive = positive,
                            alpha = alpha_best).fit(X_train.values, y_train)
            #print(X_train.shape)
            #print(X_train.iloc[:3, :3])
            #clf = clf.fit(X_train.values, y_train)
            #print(clf.score(X_train.values, y_train))
            fit_c = alpha_best#clf.alpha_
            if verbose:
                print('Best regularization: ', fit_c)
            # find p-values
            coefs =  clf.coef_
            inter = clf.intercept_
            #get regularization parameter
            if dep_var!='all':

                #coefs = np.array(list(inter)+list(coefs))
                #coefs = get_coefs(X_train, y_train, fit_c)
                pvals[g] = my_linear_pvals(clf.coef_[0], X_train.values, y_train,fit_c)
            #print(clf.pvalues())

            y_train_pred = clf.predict(X_train.values)
            y_test_pred = clf.predict(X_test.values)

            if cross_validate:
                val_acc = max(mean_scrs_valid)
            else:
                if metr == 'r2':
                    val_acc = clf.score(X_train.values, y_train)
                elif metr == 'pearson':
                    val_acc = scipy.stats.pearsonr(np.ravel(y_train_pred), np.ravel(y_train))[0]

            if metr == 'r2':
                test_acc = clf.score(X_test.values, y_test)
            elif metr == 'pearson':
                test_acc = scipy.stats.pearsonr(np.ravel(y_test_pred), np.ravel(y_test))[0]
            y_train_preds[g] = y_train_pred
            y_test_preds[g] = y_test_pred
            baseline = 0#np.sum(y_test==1)/len(y_test)
            #print(val_acc, g, valid_scores)
            valid_scores[g] = val_acc
            test_scores[g] = test_acc
            base_scores[g] = baseline
            diff_scores[g] = val_acc-baseline
            if verbose:
                print('Validation accuracy: ', val_acc)
                print('Test accuracy: ', test_acc)
            #print('Percent test positive: ', baseline)
            #print(clf.coef_)
            #plot_precision_recall_curve(clf, X_test, y_test, ax = ax, label = 'separate')
            if motif_mat is not None:
                coefs = pd.DataFrame(np.array(list(clf.coef_)).reshape(1,-1), 
                                     columns = X_train.columns)
                                 #columns = motif_mat.columns.get_level_values(0)[to_leave])
            else:
                coefs = pd.DataFrame(np.array(list(clf.coef_)).reshape(1,-1),  
                                     columns = X_train.columns)
            #print(coefs[coefs>0].T.dropna())
            if (test_acc-baseline)>thres:
                #print('adding')
                all_coeffs[g] = coefs
        except:
            if verbose:
                print('Something went wrong')
            continue
    return all_coeffs, valid_scores, test_scores, base_scores, diff_scores, pvals, y_train_preds, y_test_preds