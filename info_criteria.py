import numpy as np
import time
from pyuoi.utils import log_likelihood_glm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.special import xlogy
import scipy
import pdb

# Generalized information criterion with arbitrary penalty
def GIC(y, y_pred, model_size, penalty):

    y = y.ravel()
    y_pred = y_pred.ravel()

    ll = log_likelihood_glm('normal', y, y_pred)
    return -2 * ll + penalty * model_size

# Extended BIC (found in literature)
def eBIC(y, y_pred, n_features, model_size):

    n_samples = y.size

    # Kappa is a hyperparameter tuning the strength of the effect (1 -> no effect)
    kappa = 0.5

    eBIC = GIC(y, y_pred, model_size, np.log(n_samples)) + \
    n_features *  2 * (1 - kappa) * np.log(float(scipy.special.binom(n_features, model_size)))

    return eBIC

# modified BIC penalty with some prior on the model size
def mBIC(y, y_pred, model_size, sparsity_prior):

    # make sure sparsity prior is epsilon less than 1
    if sparsity_prior == 1:
        sparsity_prior = 0.9999

    mBIC =  BIC(y, y_pred, model_size) + 2 * model_size * np.log(1/sparsity_prior - 1)

    return mBIC

# mixture coding MDL criteria (eq. 34 in Hansen and Yu)
# k : number of features in model 
# n : number of samples
def gMDL(y, y_pred, k):
    
    # Doesn't seem to be able to handle the k = 0 case
    if k == 0:
        return np.inf, np.nan
    n = y.size
    threshold = k/n
    r2 = r2_score(y, y_pred)

    RSS = np.sum((y - y_pred)**2)
    S = RSS/(n - k)
    F = (np.sum(y**2) - RSS)/(k * S)

    if r2 > threshold:
        penalty = k/2 * np.log(F) + np.log(n)
        gMDL = n/2 * np.log(S) + k/2 * np.log(F) + np.log(n)
    else:
        penalty = 1/2 * np.log(n)
        gMDL = n/2 * np.log(np.sum(y**2)/n) + 1/2 * np.log(n)

    return gMDL, penalty

# Empirical bayesian procedure (Calibration and Empirical Bayes 
# Variable Selection)
# Actual form of the penalty taken from Adaptive Bayesian variable selection criteria for GLM
def empirical_bayes(X, y, y_pred, ssq_hat, beta):

    n, p = X.shape
    beta = beta.ravel()
    y = y.ravel()
    support = np.array(beta).astype(bool)

    # Paper provides closed form expression
    # Using the conditional marginal likelihood criterion
    q = np.count_nonzero(beta)
    
    ll = log_likelihood_glm('normal', y, y_pred)

    if q > 0:
        support = (beta !=0).astype(bool)
        Tgamma = beta[support].T @ X[:, support].T @ X[:, support] @ beta[support]/ssq_hat

        R = -2 * (xlogy(p - q, p - q) + xlogy(q, q))

        if np.divide(Tgamma, q) > 1:

            B = q + q * np.log(Tgamma) - xlogy(q, q)

            CCML = -2 * ll + B + R 

        else:
            B = Tgamma
            CCML = -2 * ll + Tgamma + R

        return CCML, B, R
    else:
        # Do not give the opportunity to select support wiht 0 coefficients
        return np.inf, 0, 0

# Full Bayes factor
def full_bayes_factor(y, y_pred, n_features, model_size, sparsity_prior, penalty):

    y = y.ravel()
    y_pred = y_pred.ravel()

    n_samples = y.size

    # Log likelihood
    ll = log_likelihood_glm('normal', y, y_pred)

    # Regularization Penalty (prior)
    p1 = 2 * penalty * model_size

    # Normal BIC penalty
    BIC = model_size * np.log(n_samples)

    # Second order Bayes factor approximation
    RSS = np.sum((y - y_pred)**2)
    BIC2 = n_samples**3/(2 * RSS*3)

    # Term arising from normalization
    BIC3 = model_size * np.log(2 * np.pi)

    # If provided with a list of sparsity estimates, we are specifying
    # a beta hyperprior, and need to integrate over it correspondingly
    if not np.isscalar(sparsity_prior):
        M_k = beta_binomial_model(sparsity_prior, n_features, model_size)
    else:
        if sparsity_prior == 1:
            sparsity_prior = 0.999

        # Model probability prior
        M_k = scipy.special.binom(n_features, model_size) * \
              sparsity_prior**model_size * (1 - sparsity_prior)**(n_features - model_size)

    # If the model probability evaluates to 0, set it to a very small but finite value to 
    # avoid blowups in the log
    if M_k == 0:
        M_k = 1e-9

    P_M = 2 * np.log(M_k)

#    bayes_factor = 2 * ll - BIC - BIC2 + BIC3 - p1 + P_M

    return ll, p1, BIC, BIC2, BIC3, M_k, P_M


# Return a posterior estimate for the binomial parameter given a 
# beta-distribution prior on estimates of that parameter 
def beta_binomial_model(x, n, k):

    # Treat warnings like errors
    np.seterr(all='raise')

    # drop all entries that are 0 
    x = x[x != 0]

    # After this point, it may be the case that x is empty or has only a single
    # element. In this case, we return a p of 0.
    if len(x) < 2:
        return 0
    else:
        # try:
        #     # Fit the parameterqs of the beta distribution
        #     a, b, _, _ = scipy.stats.beta.fit(x, floc = 0, fscale = 1)
        #     p = scipy.special.binom(n, k) * \
        #         scipy.special.beta(k + a, n - k + b)/scipy.special.beta(a, b)
        # except:
            
        # Too many edge cases, so for now just average over sparsity estimates
        p = np.mean(x)
        return p
