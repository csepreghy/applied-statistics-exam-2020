import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

from iminuit import Minuit
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chisquare, normaltest
import math

from scipy import stats
from scipy.stats import binom, poisson, norm
from scipy.integrate import quad, simps
from scipy.optimize import minimize

from plotify import Plotify

plotify = Plotify()

a_constant = 149597870700
orbital_days = [87.77, 224.70, 686.95, 4332.62, 10759.2]
a_vals = np.array([0.389, 0.724, 1.524, 5.20, 9.51]) * a_constant
a_errors = np.array([0.011, 0.020, 0.037, 0.13, 0.34]) * a_constant

G = 7.5
G_error = 1

xvals = np.array([1, 2, 3, 4, 5])

# fig, ax = plotify.get_figax(figsize=(8,6), use_grid=True)

# ax.scatter(xvals, a_vals, c=plotify.c_orange)
# ax.errorbar(xvals, a_vals, yerr=a_errors, fmt='o', c=plotify.c_orange, capsize=2.5, label="Measurement with Uncertainty")
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.set_title("Hubble Constant Measurements")
# ax.set_xlabel("Measurement Number")
# ax.set_ylabel("Value / Uncertainty")
# plt.legend(facecolor="#282D33")
# plt.show()

def keplers_law(C, T):
	a = C * T**(2/3)
	return a


def minimize_chi2(C, a_vals, uncertainties, orbital_days):
	chi2_value, chi2_pval = chi2_errors(a_vals, uncertainties, orbital_days, C, 5)
	print(f'chi2_pval = {chi2_pval}')
	print(f'chi2_value = {chi2_value}')
	return chi2_value

def chi2(observed_values, orbital_days, C):
	chi2_value = 0
	ndof = len(observed_values) - 1

	expected_values = []

	for day in orbital_days:
		a = keplers_law(C, day)
		expected_values.append(a[0])

	for observed_value, expected_value in zip(observed_values, expected_values):
		chi2_value += (observed_value - expected_value)**2 / expected_value
	
	p_chi2 = stats.chi2.sf(chi2_value, ndof)

	return chi2_value, p_chi2

def chi2_errors(values, uncertainties, orbital_days, C, n_parameters):
	ndof = 3
	chi2_value = 0

	expected_values = []

	for day in orbital_days:
		a = keplers_law(C, day)
		expected_values.append(a[0])

	for observed_value, expected_value, uncertainty in zip(values, expected_values, uncertainties):
		chi2_value += (observed_value - expected_value)**2 / uncertainty**2

	p_chi2 = stats.chi2.sf(chi2_value, ndof)

	return chi2_value, p_chi2


x0 = 2e9
res = minimize(minimize_chi2, x0, method='Nelder-Mead', tol=1e-6, args=(a_vals, a_errors, orbital_days))

print(f'res.x = {res.x}')
expected_values = []

for day in orbital_days:
	a = keplers_law(res.x, day)
	expected_values.append(a[0])

print(f'a_vals = {a_vals}')
print(f'expected_values = {expected_values}')

fig, axxx = plotify.get_figax()
axxx.errorbar(xvals, a_vals, yerr=a_errors, fmt='o', c=plotify.c_orange, markersize='5', capsize=2.5, label="Measurement with Uncertainty", alpha=0.5)
axxx.scatter(xvals, a_vals, c=plotify.c_orange, s=5)
axxx.scatter(xvals, expected_values, c=plotify.c_blue, s=15)
axxx.xaxis.set_major_locator(ticker.MultipleLocator(1))
axxx.set_title("Planet Orbital Distances")
axxx.set_xlabel("Planet Number")
axxx.set_ylabel("Value / Uncertainty")
plt.savefig(('plots/' + 'kepler'), facecolor=plotify.background_color, dpi=180)


plt.legend(facecolor="#282D33")
plt.show()
