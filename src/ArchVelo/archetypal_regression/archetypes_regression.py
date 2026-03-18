import numpy as np
import scipy
import pandas as pd
import anndata
import scanpy as sc
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
import sys

# Append path if not already present
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
                      atac_layer='X_magic',
                      feature_type='soft_clust'):  
    
    # Optimize index querying
    rel_peaks = peaks_to_genes.query('gene == "'+str(g)+'"').index
    rel_peaks = rel_peaks.intersection(train_df.var.index).intersection(test_df.var.index)
    
    # --- BRANCH 1: Standard Peak Features ---
    if feature_type != 'soft_clust':
        # Optimize: Slice directly to numpy/sparse to avoid pandas overhead
        X_train_raw = train_df[:, rel_peaks].layers[atac_layer]
        X_test_raw = test_df[:, rel_peaks].layers[atac_layer]

        # Handle sparse conversion efficiently
        if scipy.sparse.issparse(X_train_raw): X_train_raw = X_train_raw.toarray()
        if scipy.sparse.issparse(X_test_raw): X_test_raw = X_test_raw.toarray()
        
        X_train_final = X_train_raw.astype(float)
        X_test_final = X_test_raw.astype(float)
        
        cols = rel_peaks
        
        cell_by_mot_train = pd.DataFrame(X_train_final, index=train_df.obs.index, columns=cols)
        cell_by_mot_test = pd.DataFrame(X_test_final, index=test_df.obs.index, columns=cols)
        
        # Keep all columns
        to_leave = [True] * len(rel_peaks)
        cell_by_mot_train = cell_by_mot_train.loc[:, to_leave]
        cell_by_mot_test = cell_by_mot_test.loc[:, to_leave]

        return cell_by_mot_train, cell_by_mot_test

    # --- BRANCH 2: Soft Clustering (Components Scaled by Peak Weights) ---
    else:
        rel_cells_train = train_df.obs.index
        rel_cells_test = test_df.obs.index

        # C: (Cells x Comps)
        C_train = cell_train_comps.loc[rel_cells_train, :].values.astype(float)
        C_test = cell_test_comps.loc[rel_cells_test, :].values.astype(float)
        
        # W: (Comps x Peaks)
        W = peak_comps.loc[:, rel_peaks].values.astype(float)
        
        # --- MATH OPTIMIZATION ---
        # Original: (Cells x 1) outer product (1 x Peaks) -> groupby sum
        # New: Sum the peak weights first, then scale the cells.
        # W_sum: (Comps,)
        W_sum = W.sum(axis=1)
        
        # Multiply cell components by the aggregated peak weights via broadcasting
        X_train = C_train * W_sum
        X_test = C_test * W_sum
        
        # Recreate the dataframes with original column names
        num_comps = C_train.shape[1]
        col_names = ['Peak_clust_' + str(i) for i in range(num_comps)]
        
        cell_by_mot_train = pd.DataFrame(X_train, index=rel_cells_train, columns=col_names)
        cell_by_mot_test = pd.DataFrame(X_test, index=rel_cells_test, columns=col_names)
        
        to_leave = [True] * num_comps
        cell_by_mot_train = cell_by_mot_train.loc[:, to_leave]
        cell_by_mot_test = cell_by_mot_test.loc[:, to_leave]

        return cell_by_mot_train, cell_by_mot_test

def rna_target(g, rna_data, rna_layer='Ms', verbose=False):
    if verbose:
        print('RNA target, from layer ' + rna_layer)
    y = rna_data[:, g].layers[rna_layer]
    return y
                    
def generate_train_test_data(g,
                      rna_data, 
                      train_df, 
                      test_df,
                      rna_layer='Ms', 
                      atac_layer='X_magic',
                      peaks_to_genes=None,
                      feature_type='soft_clust',                      
                      cell_train_comps=None,
                      cell_test_comps=None,
                      peak_comps=None,
                      train_ind_orig=None, 
                      test_ind_orig=None,               
                      verbose=False):
    
    # 1. Generate Features (X)
    X_train, X_test = generate_features(g,                                                                            
                      peaks_to_genes,
                      train_df,
                      test_df,
                      cell_train_comps,
                      cell_test_comps,
                      peak_comps,
                      atac_layer=atac_layer,
                      feature_type=feature_type)
    
    # 2. Generate Target (y) - Strict RNA mode
    full_y = np.array(rna_target(g, rna_data, rna_layer=rna_layer, verbose=verbose))
    
    # 3. Split based on provided indices
    y_train = full_y[train_ind_orig]
    y_test = full_y[test_ind_orig]
    
    return X_train, y_train, X_test, y_test

def normalize_features(X_train, X_test, stand_feat='z-score'):
    if stand_feat == 'z-score':
            X_train = standardize(X_train).fillna(0)
            X_test = standardize(X_test).fillna(0)
    elif stand_feat == 'minmax':
            X_train = minmax(X_train)
            X_test = minmax(X_test)  
    return X_train, X_test
    
def atac_regression_with_components(
                        genes=None, 
                        XC_train=None, 
                        XC_test=None,
                        S=None, 
                        cross_validate=True,
                        l=None,
                        XC_train_folds=None, 
                        XC_test_folds=None,
                        S_folds=None, 
                        rna_data=None,
                        train_df=None, 
                        test_df=None, 
                        train_ind_orig=None, 
                        test_ind_orig=None,
                        positive=False,
                        reg_type='ridge',
                        alphas=np.logspace(-7, 5, 50),
                        alpha_to_use=0.1, 
                        svr_kernel='rbf',
                        neural_solver = 'adam',
                        activation='relu',
                        hidden_layer_sizes = (16,8),
                        early_stopping = False,
                        fit_intercept=True,
                        peaks_to_genes=None,
                        feature_type='soft_clust',
                        metr='r2',
                        stand_feat=False, 
                        rna_layer='Ms',
                        atac_layer='pearson',
                        random_state = 42,
                        verbose=False):
    
    all_coeffs = {}; valid_scores = {}; test_scores = {}
    base_scores = {}; diff_scores = {}; pvals = {}
    y_train_preds = {}; y_test_preds = {}
    scrs_train_out = {}; scrs_valid_out = {}

    for g in genes:
        if verbose: print(g)
        try:
            scrs_valid = {}
            if cross_validate:
                if l is None: return None
                arr_l = np.array(l)
                scrs_train = {}
                
                # Pre-calculate data for all 3 folds
                folds_data = []
                for fld in [0, 1, 2]:
                    X_tr, y_tr, X_te, y_te = generate_train_test_data(
                            g=g, rna_data=rna_data, rna_layer=rna_layer,
                            atac_layer=atac_layer,                                                   
                            feature_type=feature_type, peaks_to_genes=peaks_to_genes,
                            cell_train_comps=XC_train_folds[fld], cell_test_comps=XC_test_folds[fld], 
                            peak_comps=S_folds[fld], 
                            train_df=train_df[arr_l!=fld, :], test_df=train_df[arr_l==fld, :], 
                            train_ind_orig=list(np.array(train_ind_orig)[arr_l!=fld]), 
                            test_ind_orig=list(np.array(train_ind_orig)[arr_l==fld]),
                            verbose=False)
                    #print('1')
                    if stand_feat: X_tr, X_te = normalize_features(X_tr, X_te, stand_feat=stand_feat)
                    folds_data.append((X_tr, y_tr, X_te, y_te))

                if reg_type == 'nystrom':
                    transformed_folds = []
                    for fld in [0, 1, 2]:
                        X_tr, y_tr, X_te, y_te = folds_data[fld]
                        
                        # 300 components is a good balance. Reduce to 100 for more speed if needed.
                        nys = Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=random_state)
                        
                        # Transform and overwrite
                        X_tr_new = nys.fit_transform(X_tr.values)
                        X_te_new = nys.transform(X_te.values)
                        
                        # Keep y as is, wrap X back in DF to keep downstream logic compatible
                        # (Though Ridge accepts numpy arrays fine)
                        transformed_folds.append((pd.DataFrame(X_tr_new), y_tr, pd.DataFrame(X_te_new), y_te))
                    
                    # Swap the data source to the transformed one
                    folds_data = transformed_folds

                scrs_train = {0:{}, 1:{}, 2:{}}
                scrs_valid = {0:{}, 1:{}, 2:{}}

                current_grid = alphas
                if reg_type == 'kernel':
                    current_grid = np.logspace(-7, 5, 10) 
                    eps = np.std(folds_data[0][1]) * 0.1
                elif reg_type == 'neural':
                    current_grid = np.logspace(-5, 0, 6)


                # Loop over grid
                for val in current_grid:
                    # Define model
                    if reg_type == 'lasso': model = Lasso(fit_intercept=fit_intercept, positive=positive, alpha=val, warm_start=True) 
                    elif reg_type in ['ridge', 'nystrom']: 
                        model = Ridge(fit_intercept=fit_intercept, positive=positive, alpha=val)
                    elif reg_type == 'kernel': model = SVR(kernel= svr_kernel, C=val, epsilon=eps)
                    elif reg_type == 'neural': model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                                                                    activation = activation, 
                                                                    alpha=val, 
                                                                    max_iter=500, 
                                                                    solver=neural_solver,
                                                                    early_stopping=early_stopping,
                                                                   random_state = random_state)

                    # Loop over pre-calculated folds
                    for fld in [0, 1, 2]:
                        X_train_v, y_train_v, X_test_v, y_test_v = folds_data[fld]
                        y_train_flat = np.ravel(y_train_v)

                        # --- DEBUG: Safety Check ---
                        if reg_type == 'nystrom' and X_train_v.shape[1] < 100:
                            raise RuntimeError(f"Nystrom failed! Features are still shape {X_train_v.shape}. Expected ~300.")
                        # ---------------------------
                        
                        model.fit(X_train_v.values, y_train_flat)
                        
                        if metr == 'r2':
                            scrs_train[fld][val] = model.score(X_train_v.values, y_train_v)
                            scrs_valid[fld][val] = model.score(X_test_v.values, y_test_v)
                        else: # Pearson
                            scrs_train[fld][val] = scipy.stats.pearsonr(y_train_flat, np.ravel(model.predict(X_train_v.values)))[0]
                            scrs_valid[fld][val] = scipy.stats.pearsonr(np.ravel(y_test_v), np.ravel(model.predict(X_test_v.values)))[0]
                            #print('Fold: ', fld)
                            #print('y_test: ', np.ravel(y_test_v))
                            #print('pred: ', np.ravel(model.predict(X_test_v.values)))
                
                sc_tr_df = pd.DataFrame(scrs_train)
                sc_val_df = pd.DataFrame(scrs_valid)
                scrs_train_out[g], scrs_valid_out[g] = sc_tr_df, sc_val_df
                # print(sc_val_df)
                # print(sc_val_df.mean(1))
                # print(sc_val_df.mean(1).idxmax())

                alpha_best = sc_val_df.mean(1).idxmax()
                if np.isnan(alpha_best):
                    alpha_best = alpha_to_use
                
                if verbose:
                    print('Best regularization: ', alpha_best)
            else:
                alpha_best = alpha_to_use

            # Final Fit
            X_train, y_train, X_test, y_test = generate_train_test_data(
                            g=g, rna_data=rna_data, rna_layer=rna_layer,
                            atac_layer=atac_layer,                                                 
                            feature_type=feature_type, peaks_to_genes=peaks_to_genes,
                            cell_train_comps=XC_train,
                            cell_test_comps=XC_test, peak_comps=S,
                            train_df=train_df, test_df=test_df, 
                            train_ind_orig=train_ind_orig, 
                            test_ind_orig=test_ind_orig, 
                            verbose=False)
            
            if stand_feat: X_train, X_test = normalize_features(X_train, X_test, stand_feat=stand_feat)
            
            # --- FINAL FIT OPTIMIZATION ---
            # 1. If Nystrom, transform the data first, then pretend it's just Ridge
            if reg_type == 'nystrom':
                nys = Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=random_state)
                # Wrap in DataFrame so .values attribute works in the standard block below
                X_train = pd.DataFrame(nys.fit_transform(X_train.values), index=X_train.index)
                X_test = pd.DataFrame(nys.transform(X_test.values), index=X_test.index)
                
            
            # 2. Standard Model Definition (Now handles Nystrom-transformed data too)
            if reg_type == 'lasso': 
                clf = Lasso(fit_intercept=fit_intercept, positive=positive, alpha=alpha_best)
            elif reg_type == 'ridge' or reg_type == 'nystrom': 
                clf = Ridge(fit_intercept=fit_intercept, positive=positive, alpha=alpha_best)
            elif reg_type == 'kernel': 
                clf = SVR(kernel=svr_kernel, C=alpha_best, epsilon=np.std(y_train)*0.1)
            elif reg_type == 'neural': 
                clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                                   activation=activation, 
                                   alpha=alpha_best, 
                                   max_iter=1000, 
                                   solver=neural_solver, 
                                   early_stopping=early_stopping, 
                                   random_state=random_state)

            clf.fit(X_train.values, np.ravel(y_train))
            y_train_pred, y_test_pred = clf.predict(X_train.values), clf.predict(X_test.values)

            # Scores
            val_acc = sc_val_df.mean(1).max() if cross_validate else (clf.score(X_train.values, y_train) if metr=='r2' else scipy.stats.pearsonr(np.ravel(y_train_pred), np.ravel(y_train))[0])
            test_acc = clf.score(X_test.values, y_test) if metr=='r2' else scipy.stats.pearsonr(np.ravel(y_test_pred), np.ravel(y_test))[0]
            
            y_train_preds[g], y_test_preds[g] = y_train_pred, y_test_pred
            valid_scores[g], test_scores[g] = val_acc, test_acc
            base_scores[g] = 0 
            diff_scores[g] = val_acc - base_scores[g]

            if verbose:
                print('Validation accuracy: ', val_acc)
                print('Test accuracy: ', test_acc)
            
            if reg_type in ['ridge', 'lasso']:
                pvals[g] = my_linear_pvals(clf.coef_, X_train.values, y_train, alpha_best)
                all_coeffs[g] = pd.DataFrame(clf.coef_.reshape(1,-1), columns=X_train.columns)

        except Exception as e:
            if verbose: print(f"Error {g}: {e}")
            continue
            
    return all_coeffs, valid_scores, test_scores, base_scores, diff_scores, pvals, y_train_preds, y_test_preds, scrs_train_out, scrs_valid_out