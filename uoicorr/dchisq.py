import numpy as np
from scipy.special import gamma, gammaln
import mpmath as mp
import scipy.integrate as integrate
import sympy
import pdb

class DChiSq(): 

    # Distribution of the weighted difference between two chi squared
    # random variables

    # alpha Chi^2(n) - beta Chi^2(m)

    # Lifted from https://projecteuclid.org/download/pdf_1/euclid.aoms/1177699531

    def __init__(self, alpha, beta, m, n):

        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.m = m

    # Psi function
    def psi_(self, a, b, x):
        # This does not return a number but rather an mpmath string
        return mp.hyperu(a, b, x)

    # Upper bound on psi_ using cauchy-schwarz
    def ub_psi_(self, a, b, x):

        return 1/mp.gamma(a) * \
               mp.sqrt(mp.power(2, a) * mp.power(x, -a) * mp.gamma(a) * mp.exp(x/2) * mp.expint(1 + a - b, x/2))

    # Normalization constant
    def norm_const(self):

        c = mp.power(mp.power(2, 0.5 * (self.n + self.m)), -1)
        return c

    # Return the PDF evaluated at x (scalar or ndarray)
    def PDF(self, x):

        if np.isscalar(x):
            x = np.array([x])

        p = self.norm_const() * np.ones(x.size)

        # Negate all x that are less than 0
        x = np.abs(x)

        # Use efficient numpy to evaluate the basic functions
        p = np.multiply(p, np.power(x, self.m + self.n - 2) * np.exp(-x/(2 * self.alpha)))        

        # List comprehension to evaluate psi
        psi = np.array([self.psi_(self.n/2, (self.m + self.n)/2, 
                        (self.alpha + self.beta)/(2 * self.alpha * self.beta) * xx) 
                        for xx in x])

        p = np.multiply(p, psi)

        if p.size == 1:
            return p[0]
        else:
            return p

    # Return the log PDF (usually more manageable) 

    # The PDF is unimodal --> once we hit a threshold, stop evaluating as 
    # we then tend to get in trouble with negative values 
    # thresh: log threshold
    def logPDF(self, x, thresh = -50):

        if np.isscalar(x):
            x = np.array([x])

        xposmask = x >= 0
        xpos = x[xposmask]

        xnegmask = x < 0
        xneg = x[xnegmask]

        #### Positive part of the PDF
        if len(xpos) > 0:

            p1 = float(mp.log(self.norm_const()) -  mp.log(mp.gamma(self.m/2))) * np.ones(xpos.size)

            # Basic functions
            p1 += (self.m + self.n - 2)/2 * np.log(xpos) - xpos/(2 * self.alpha)

            # Keep track of diffs
            pdiffs = np.zeros(x.size - 1)

            for i, xx in enumerate(xpos):
#                try:
                psi_ = self.psi_(self.n/2, 
                                 (self.m + self.n)/2, 
                                 (self.alpha + self.beta)/(2 * self.alpha * self.beta) * xx)
                p1[i] = p1[i] + psi_                
#                except:
#                    p1[i] = np.nan             
            #     if i > 0:
            #         pdiffs[i - 1] = p1[i] - p1[i -1]

            #         if pdiffs[i - 1] < 0 and p1[i] < thresh:

            #             break 

            # # Set anything leftover if the threshold has been crossed to 0

            # if i < xpos.size - 1:

            #     p1[i:] = np.nan
        else:
            p1 = []

        #### Negative part of the PDf
        if len(xneg) > 0:
            xneg = np.abs(xneg)
            p2 = float(mp.log(self.norm_const() / mp.gamma(self.n/2))) * np.ones(xneg.size)

            # Basic functions
            p2 += (self.m + self.n - 2) * np.log(xneg) - xneg/(2 * self.beta)

            for i, xx in enumerate(xpos):
                try:
                    p2[i] += float(mp.log(self.psi_(self.m/2, 
                                                    (self.m + self.n)/2, 
                                                    (self.alpha + self.beta)/(2 * self.alpha * self.beta) * xx)))
                except:
                    p2[i] = np.nan             


        else:
            p2 = []

        p = np.zeros(x.size)
        p[xposmask] = p1
        p[xnegmask] = p2

        if p.size == 1:
            return p[0]
        else:
            return p        

    def char_fn(self, t, alpha = None, beta = None, n = None, m = None):

        # Product of chi squared characteristic functions, rescaled by factors to adjust
        # for the fact that we have the weighted difference. Also note the complex conjugation

        return mp.mpc((1 - 2 * 1j * self.alpha * t)**(-self.m/2) * (1 + 2 * self.beta * 1j * t)**(-self.n/2))


    # Calculate the mean and variance in a region surrounding the 
    def mean(self):

        return self.alpha * self.m - self.beta * self.n

    def variance(self):

        return 2 * (self.m * self.alpha**2 * self.n * self.beta**2)

    # Calculate the symbolic derivatives once up front and store them in a list
    # Calculates all derivatives of order <= n
    def gen_diffs(self, n_):
        a, b, m, n, t = sympy.symbols('a b m n t')
        self.diffs = []
        for i in range(1, n_ + 1):
            expr = sympy.diff((1 - 2 * 1j * a * t)**(-m/2) * (1 + 2 * 1j * b * t)**(-n/2), t, i)
            self.diffs.append(expr.subs({a: self.alpha, b: self.beta, m: self.m, n: self.n}))

    def series_term(self, order, omega, ub):

        # The order 1 term requires evaluation of the characteristic function
        # and not its derivatives
        t = sympy.symbols('t') 
        if order == 1:
            return mp.mpc(1/(-1j * omega)**order * \
                   (mp.exp(-1j * omega * ub) * self.char_fn(ub) - self.char_fn(0)))
        else:
            fm = sympy.lambdify(t, self.diffs[order - 2])
            return mp.mpc(1/(-1j * omega)**order * \
                   (mp.exp(-1j * omega * ub) * fm(ub) - fm(0)))

    # Use the method detailed in https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2004.1401
    # to handle highly oscillatory instantiations of the characteristic function inversion integral
    def asymptotic_expansion(self, omega):

        # Evaluate the modulus of the characteristic function
        domain = np.linspace(0, 5, 500)
        char_fn = list(map(lambda t: mp.fabs(self.char_fn(t)), domain))

        # thresh 1e-40 
        thresh_check = [domain[i] for i in range(500) if char_fn[i] < mp.mpf(1e-50)]

        # Need to extend the domain
        if len(list(thresh_check)) == 0:
            j = 1
            while len(list(thresh_check)) == 0:

                domain = np.linspace(5 * j, 5 * (j + 1), 500)
                char_fn = list(map(lambda t: mp.fabs(self.char_fn(t)), domain))
                # thresh 1e-40 
                thresh_check = [domain[i] for i in range(500) if char_fn[i] < mp.mpf(1e-50)]
                j += 1

        cutoff = thresh_check[0]

        # Generate the derivatives for the asymptotic expansion
        order = 6

        if not hasattr(self, 'diffs'):
            self.gen_diffs(order)

        # Evaluate the expansion
        asym_series = mp.matrix(order, 1)
        for i in range(1, order + 1):
            asym_series[i - 1] = self.series_term(i, omega, cutoff)

        # Sum up and take the real part
        # We do not multiply by (-1) because the fact that we have g(x) = -x cancels this
        return mp.re(mp.fsum(asym_series))

    # Find the roots of the characteristic function to facillitate oscillatory quadrature
    # integration

    # fn: gil-pelaez integrand
    # ub: Cutoff of integration domain

    def find_roots(self, fn, ub):

        # Start with a coarse evaluation and fine tune the mesh until the number of
        # detected sign changes no longer increases

        init = True
        cont = False
        nchangepnts = 0
        i = 1
        while init or cont:
            init = False

            domain = np.linspace(0, ub, 1000 * i)
            cf0 = list(map(fn, domain))

            signs = np.array(list((map(lambda t: float(mp.sign(t)), cf0))))
            dsign = ((np.roll(signs, 1) - signs) != 0).astype(int)
            dsign[0] = 0
            changepnts = np.nonzero(dsign)[0]

            if changepnts.size > nchangepnts:
                cont = True
                nchangepnts = changepnts.size
            else:
                cont = False

        halfchangepnts = ((changepnts[1:] + changepnts[:-1])/2).astype(int)
        partpnts = np.zeros(halfchangepnts.size + 2, dtype=int)
        partpnts[-1] = (len(cf0) - 1)
        partpnts[1:-1] = halfchangepnts

        roots = []

        # Search for the root within each pair of partpnts
        for i in range(len(partpnts[:-1])):    
            roots.append(mp.findroot(fn, (domain[partpnts[i]], domain[partpnts[i+1]]), 
                                     solver='anderson'))

        return roots

    def find_cutoff(self, thresh):
        # Find a cutoff to do finite integration

        # Evaluate the modulus of the characteristic function
        domain = np.linspace(0, 5, 500)
        char_fn = list(map(lambda t: mp.fabs(self.char_fn(t)), domain))

        # thresh 1e-40 
        thresh = 1e-30
        thresh_check = [domain[i] for i in range(500) if char_fn[i] < mp.mpf(thresh)]

        # Need to extend the domain
        if len(list(thresh_check)) == 0:
            j = 1
            while len(list(thresh_check)) == 0:

                domain = np.linspace(5 * j, 5 * (j + 1), 500)
                char_fn = list(map(lambda t: mp.fabs(self.char_fn(t)), domain))
                # thresh 1e-40 
                thresh_check = [domain[i] for i in range(500) if char_fn[i] < mp.mpf(thresh)]
                j += 1

        cutoff = thresh_check[0]

        return cutoff

    # Calculate the PDF via numerical inversion of the characteristic function
    def nPDF(self, x):

        p = np.zeros(x.size)
        for i, xx in enumerate(x):

            gil_pelaez = lambda t: mp.re(self.char_fn(t) * mp.exp(-1j * t * xx))

            cutoff = self.find_cutoff(1e-30)
            # Instead of finding roots, break up quadrature into degrees proportional to the 
            # expected number of oscillations of e^(i xx t) within t = [0, cutoff]
            nosc = cutoff/(1/max(10, np.abs(xx - self.mean())))
#            roots = self.find_roots(gil_pelaez, cutoff)
#            if np.abs(xx - self.mean()) < 3 * np.sqrt(self.variance()):

            I = mp.quad(gil_pelaez, np.linspace(0, cutoff, nosc), maxdegree=10)
#            I = mp.quadosc(gil_pelaez, (0, cutoff), zeros=roots)

#            else:
                # For now, do not trust any results out greater than 3sigma

#                I = 0

            # if np.abs(xx - self.mean()) >= 2 * np.sqrt(self.variance()):

#            I = self.asymptotic_expansion(xx)

            p[i] = 1/np.pi * float(I)
            print(i)
        return p

    # Calculate the CDF via numerical inversion of the characteristic function
    def nCDF(self, x):

        if np.isscalar(x):
            x = np.array([x])

        p = np.zeros(x.size)

        for i, xx in enumerate(x):

            gil_pelaez = lambda t: mp.im(self.char_fn(t) * mp.exp(-1j * t * xx))/t

            cutoff = self.find_cutoff(1e-30)
            # Instead of finding roots, break up quadrature into degrees proportional to the 
            # expected number of oscillations of e^(i xx t) within t = [0, cutoff]
            nosc = cutoff/(1/max(10, np.abs(xx - self.mean())))


            I = mp.quad(gil_pelaez, np.linspace(0, cutoff, nosc), maxdegree=10)
            p[i] = float(1/2 - 1/mp.pi * I)

        if p.size == 1:
            return p[0]
        else:
            return p        

    def MCCDF_(self, x, n_samples = 1000000):

        # Use Monte Carlo samples to approximate the CDF

        threshold_count = 0

        for _ in range(n_samples):

            x1 = np.random.chisquare(self.m)
            x2 = np.random.chisquare(self.n)

            sample = self.alpha * x1 - self.beta * x2 

            if sample <= x:
                threshold_count += 1

        return float(threshold_count)/float(n_samples)

    # Return the CDF evaluated at x (scalar or ndarray)
    def CDF(self, x, method='MC'):

        if np.isscalar(x):
            x = np.array([x])

        c = np.zeros(x.size)
        for i, xx in enumerate(x):
            c[i] = self.MCCDF_(xx)

        return c

        # Return the inverse CDF evaluated at x:
    def invCDF(self, x):
        pass
