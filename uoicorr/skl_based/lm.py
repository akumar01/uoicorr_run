import numpy as np 
import pdb
from sklearn.linear_model import ElasticNet


class EN_Grid()

	def __init__(self, l1_ratios, l1_params, fit_intercept=False):

		self.l1_ratios = l1_ratios
		self.l1_params = l1_params
		self.fit_intercept = fit_intercept

	def fit(X, y):

		n, p = X.shape()
		coefs = np.zeros((len(self.l1_ratios), len(self.l1_params), p))


		# Utilize warm start
		en = ElasticNet(fit_intercept=self.fit_intercept, warm_start=True)

		for i1, l1_ratio in enumerate(self.l1_ratios):
			for i2, l1_param in enumerate(self.l1_params):

				en.set_params(alpha=l1_param, l1_ratio=l1_ratio)
				en.fit(X, y.ravel())
				coefs[i1, i2, :] = en.coef_.ravel()

		self.coef_ = coefs
