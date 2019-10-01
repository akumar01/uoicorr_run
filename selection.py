import numpy as np
import pdb
import itertools
import time
from mpi_utils.ndarray import Gatherv_rows, Bcast_from_root
from utils import selection_accuracy
from info_criteria import GIC, eBIC, gMDL, empirical_bayes
from aBIC import aBIC, mBIC
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class Selector():

    def __init__(self, selection_method = 'BIC'): 

        self.selection_method = selection_method

    def select(self, solutions, reg_params, X, y, true_model, intercept=0):

        # Common operations to all
        n_samples, n_features = X.shape

        y_pred = solutions @ X.T + intercept

        # Deal to the appropriate sub-function based on 
        # the provided selection method string

        if self.selection_method in ['mBIC', 'eBIC', 'BIC', 'AIC',
                                       'gMDL', 'empirical_bayes']:
            sdict = self.selector(X, y, y_pred, solutions, 
                                  reg_params)
        elif self.selection_method == 'oracle':
        
            sdict = self.oracle_selector(solutions, reg_params, true_model)

        elif self.selection_method == 'aBIC':
            sdict = self.aBIC_selector(X, y, solutions, 
                                       reg_params, true_model)
        else:
            raise ValueError('Incorrect selection method specified')
        return sdict

    def selector(self, X, y, y_pred, solutions, reg_params):

        n_samples, n_features = X.shape
        sdict = {}
        if self.selection_method in ['AIC', 'BIC']:
            if self.selection_method == 'BIC':
                penalty = np.log(n_samples)
            if self.selection_method == 'AIC':
                penalty = 2
            
            scores = np.array([GIC(y.ravel(), y_pred[i, :], 
                               np.count_nonzero(solutions[i, :]),
                               penalty) for i in range(solutions.shape[0])])
            sidx = np.argmin(scores)
        if self.selection_method == 'mBIC':
            scores = mBIC(X, y, solutions)
            sidx = np.argmax(scores)
        elif self.selection_method ==  'eBIC':
            scores = np.array([eBIC(y.ravel(), y_pred[i, :], n_features,
                                    np.count_nonzero(solutions[i, :]))
                                    for i in range(solutions.shape[0])])
            sidx = np.argmin(scores)

        elif self.selection_method in ['gMDL', 'empirical_bayes']:

            # Fit OLS models
            OLS_solutions = np.zeros(soltions.shape)

            for i in range(solutions.shape[0]):
                support = solutions[i, :].astype(bool)
                linmodel = LinearRegression(fit_intercept=False)
                linmodel.fit(X[:, support], y)
                OLS_solutions[i, support] = linmodel.coef_

            y_pred = OLS_solutions @ X.T

            if self.selection_method == 'gMDL':
                scores = np.array([gMDL(y.ravel(), y_pred[i, :],
                            np.count_nonzero(solutions[i, :]))
                            for i in range(solutions.shape[0])])
                sidx = np.argmin(scores[:, 0])
                sdict['effective_penalty'] = scores[sidx, 1] 

            elif self.selection_method == 'empirical_bayes':
                # Properly normalize before selection
                # y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()            
                # Require a linear regression fit on the full model to estimate 
                # the noise variance:
                bfull = LinearRegression().fit(X, y).coef_.ravel() 
                y = y.ravel()
                ssq_hat = (y.T @ y - bfull.T @ X.T @ X @ bfull)/(X.shape[0] - X.shape[1])
                scores = np.array([empirical_bayes(X, y, y_pred[i, :], ssq_hat,
                                       solutions[i, :])
                                   for i in range(solutions.shape[0])])
                sidx = np.argmin(scores[:, 0])
                sdict['effective_penalty'] = scores[sidx, 1]
        # Selection dict: Return coefs and selected_reg_param
        sdict['coefs'] = solutions[sidx, :]
        sdict['reg_param'] = reg_params[sidx]
        return sdict

    def oracle_selector(self, solutions, reg_params, true_model):
        # Quickly return the best selection accuracy
        selection_accuracies = selection_accuracy(true_model.ravel(), solutions)
        sidx = np.argmax(selection_accuracies)
        sdict = {}
        sdict['coefs'] = solutions[sidx, :]
        sdict['reg_param'] = reg_params[sidx]

        return sdict

    def aBIC_selector(self, X, y, solutions, reg_params, true_model):

        oracle_penalty, bayesian_penalty, bidx, oidx, spest = \
        aBIC(X, y, solutions, true_model)

        # Selection dict: Return coefs and selected_reg_param
        sdict = {}
        sdict['coefs'] = solutions[bidx, :]
        sdict['reg_param'] = reg_params[bidx]
        sdict['oracle_coefs'] = solutions[oidx, :]
        sdict['oracle_penalty'] = oracle_penalty
        sdict['effective_penalty'] = bayesian_penalty
        sdict['sparsity_estimates'] = spest

        return sdict

class UoISelector(Selector):

    def __init__(self, uoi, selection_method = 'CV', comm=None):

        super(UoISelector, self).__init__(selection_method)
        self.uoi = uoi
        self.comm = comm
        if comm is not None:
            self.rank = comm.rank
            self.size = comm.size
        else:
            self.rank = 0
            self.size = 1

    # Perform the UoI Union operation (median)
    def union(self, selected_solutions):
        coefs = np.median(selected_solutions, axis = 0)
        return coefs

    def select(self, X, y, true_model): 
        t0 = time.time()
        # Apply UoI pre-processing
        X, y, = self.uoi._pre_fit(X, y)

        # For UoI, interpret CV selector to mean r2
        if self.selection_method == 'CV':
            sdict = self.r2_selector(X, y)
        elif self.selection_method in ['BIC', 'AIC', 'mBIC',
                                       'eBIC', 'gMDL', 'empirical_bayes']: 
            sdict = self.selector(X, y)
        elif self.selection_method == 'oracle':
            sdict = self.oracle_selector(true_model)

        elif self.selection_method == 'aBIC':
            sdict = self.aBIC_selector(X, y, true_model)
        else:
            raise ValueError('Incorrect selection method specified')

        # Apply UoI post-processing (copy and pasted)
        if self.uoi.standardize and self.rank == 0:
            sX = self.uoi._X_scaler
            sy = self.uoi._y_scaler
            sdict['coefs'] /= sX.scale_
            sdict['coefs'] *= sy.scale_

            # Don't bother to keep track of the effective penalty 
            sdict['effective_penalty'] = np.nan
            # print('Selection Method: %s, Time: %f' % (self.selection_method, time.time() - t0))
        return sdict

    def r2_selector(self, X, y):

        # UoI Estimates have shape (n_boots_est, n_supports, n_coef)
        solutions = self.uoi.estimates_
        intercepts = self.uoi.intercepts_
        boots = self.uoi.boots

        if self.comm is not None: 
            boots = self.comm.bcast(boots)
            solutions = self.comm.bcast(solutions)
            intercepts = self.comm.bcast(intercepts)

        n_boots, n_supports, n_coefs = solutions.shape
        # Distribute bootstraps across ranks
        tasks = np.arange(n_boots)
        chunked_tasks = np.array_split(tasks, self.size)
        task_list = chunked_tasks[self.rank]
        scores = np.zeros((len(task_list), n_supports))

        for i, boot in enumerate(task_list):
            # Test data
            xx = X[boots[1][boot], :]
            yy = y[boots[1][boot]]  
            y_pred = solutions[boot, ...] @ xx.T + intercepts[boot, :][:, np.newaxis]

            scores[i, :] = np.array([r2_score(yy, y_pred[j, :]) for j in range(n_supports)])

        # Gather 
        if self.comm is not None:
            scores = Gatherv_rows(scores, self.comm)

        if self.rank == 0:

            selected_idxs = np.argmax(scores, axis = 1)
            coefs = self.union(solutions[np.arange(n_boots), selected_idxs])

            # Return just the coefficients that result
            sdict = {}
            sdict['scores'] = scores
            sdict['coefs'] = coefs

        else: 
            sdict = None

        return sdict

    def selector(self, X, y):

        solutions = self.uoi.estimates_
        intercepts = self.uoi.intercepts_
        boots = self.uoi.boots
        n_boots, n_supports, n_coefs = solutions.shape

        # Need to distribute information across ranks:
        if self.comm is not None:
            boots = self.comm.bcast(boots)
            solutions = self.comm.bcast(solutions)
            intercepts = self.comm.bcast(intercepts)

        n_boots, n_supports, n_coefs = solutions.shape

        # Distribute bootstraps across ranks
        tasks = np.arange(n_boots)    

        chunked_tasks = np.array_split(tasks, self.size)
        task_list = chunked_tasks[self.rank]

        selected_coefs = np.zeros((len(task_list), n_coefs))

        for i, boot in enumerate(task_list):
            # Train data
    
            t0 = time.time()

            xx = X[boots[0][boot], :]
            yy = y[boots[0][boot]]
            n_samples, n_features = xx.shape
            y_pred = solutions[boot, ...] @ xx.T + intercepts[boot, :][:, np.newaxis]
            
            sdict_ = super(UoISelector, self).selector(xx, yy, y_pred, solutions[boot, ...], 
                                                           np.arange(n_supports)) 

            selected_coefs[i, :] = sdict_['coefs']

            # if self.selection_method == 'empirical_bayes': 
            #    print('bootstrap time: %f' % (time.time() - t0))

        # Gather selected_coefs
        if self.comm is not None:
            selected_coefs = Gatherv_rows(selected_coefs, self.comm)

        if self.rank == 0:
            coefs = self.union(selected_coefs)
            sdict = {}
            sdict['coefs'] = coefs
        else:
            sdict = None

        return sdict

    def oracle_selector(self, true_model):
        # Simply return the maximum selection accuracy available

        solutions = self.uoi.estimates_
        intercepts = self.uoi.intercepts_
        assert(np.all(intercepts == 0))
        boots = self.uoi.boots
    
        if self.comm is not None:
            boots = self.comm.bcast(boots)
            solutions = self.comm.bcast(solutions)
            intercepts = self.comm.bcast(intercepts)

        n_boots, n_supports, n_coefs = solutions.shape

        # Distribute bootstraps across ranks
        tasks = np.arange(n_boots)
        chunked_tasks = np.array_split(tasks, self.size)
        task_list = chunked_tasks[self.rank]

        selected_coefs = np.zeros((len(task_list), n_coefs))

        for i, boot in enumerate(task_list):

            sdict_ = super(UoISelector, self).oracle_selector(solutions[boot, ...], 
                                                              np.arange(n_supports),
                                                              true_model) 

            selected_coefs[i, :] = sdict_['coefs']

        # Gather 
        if self.comm is not None:
            selected_coefs = Gatherv_rows(selected_coefs, self.comm)

        if self.rank == 0:

            coefs = self.union(selected_coefs)

            # Return just the coefficients that result
            sdict = {}
            sdict['coefs'] = coefs

        else: 
            sdict = None

        return sdict

    def aBIC_selector(self, X, y, true_model):

        solutions = self.uoi.estimates_
        intercepts = self.uoi.intercepts_
        assert(np.all(intercepts == 0))
        boots = self.uoi.boots

        n_boots, n_supports, n_coefs = solutions.shape
        bselected_coefs = np.zeros((n_boots, n_coefs))
        oselected_coefs = np.zeros((n_boots, n_coefs))
        bayesian_penalties = np.zeros(n_boots)
        oracle_penalties = np.zeros(n_boots)

        sparsity_estimates = []
        for boot in range(n_boots):

            # Train data
            xx = X[boots[0][boot], :]
            yy = y[boots[0][boot]]

            sdict_ = super(UoISelector, self).aBIC_selector(xx, yy, solutions[boot, ...],
                                                            np.arange(n_supports),
                                                            true_model)
            bselected_coefs[boot, :] = sdict_['coefs'] 
            oselected_coefs[boot, :] = sdict_['oracle_coefs']
            bayesian_penalties[boot] = sdict_['bayesian_penalty']
            oracle_penalties[boot] = sdict_['oracle_penalty']
            sparsity_estimates.append(sdict_['sparsity_estimates'])

        coefs = self.union(bselected_coefs)
        oracle_coefs = self.union(oselected_coefs)
        sdict = {}
        sdict['coefs'] = coefs
        sdict['oracle_coefs'] = oracle_coefs
        sdict['effective_penalty'] = bayesian_penalties
        sdict['oracle_penalty'] = oracle_penalties
        sdict['sparsity_estimates'] = np.array(sparsity_estimates, dtype = float)

        return sdict
