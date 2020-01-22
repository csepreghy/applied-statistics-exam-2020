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

def func_gaussian_pdf(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

def chi2_gauss(params, observed_vals, xvals):
	C, mu, sigma = params

	if mu < 0: return 999999
	if sigma < 0: return 999999
	# if C < 0: return 999999

	expected_gauss_vals = np.zeros(len(observed_vals))

	print(f'params = {params}')
	
	for i, x in enumerate(xvals):
		expected_gauss_vals[i] = C * func_gaussian_pdf(x, mu, sigma)

	print(f'observed_vals = {observed_vals[0:10]}')
	print(f'expexted_gauss_vals = {expected_gauss_vals[0:10]}')
	chi2, chi2_pval = chisquare(observed_vals, expected_gauss_vals)
	print(f'chi2 = {chi2}')
	print(f'chi2_pval = {chi2_pval}')

	return chi2

# This script reads the file "data_BetaCalibration.txt" for problem 5.2 in the exam of 
# Applied Statistics 2019/20. This data files contains 4000 entries in four column, 
# where the four numbers are the initial speed estimate (beta_init), the angle with 
# respect to the beam axis (theta in radians), the energy deposited in the calorimeters 
# (E in GeV), and the time recording the data since the start of the experiment 
# (T in seconds), respectively. The script was distributed along with the exam and 
# the data file itself on the 15th of January 2020.

beta_init, theta, E, T = np.loadtxt('data/data_BetaCalibration.txt', unpack=True)
n_bins = 100
xmin = np.min(beta_init)
xmax = np.max(beta_init)
beta_init_mean = np.mean(beta_init)
beta_init_std = np.std(beta_init)

print(f'beta_init = {beta_init}')

expected_gaussian_vals = np.zeros(n_bins)
y_observed, xExp_edges = np.histogram(beta_init, bins=n_bins, range=(xmin, xmax))
xvals = np.linspace(xmin, xmax, n_bins)

x0 = [0.6, beta_init_mean, beta_init_std]
res_gauss = minimize(chi2_gauss, x0, args=(y_observed, xvals))
print(f'res_gauss gauss x = {res_gauss.x}')
print(f'res_gauss = {res_gauss}')

expected_gauss_vals = np.zeros(len(y_observed))
for i, x in enumerate(xvals):
	expected_gauss_vals[i] = res_gauss.x[0] * func_gaussian_pdf(x, res_gauss.x[1], res_gauss.x[2])

chi2_value_gaussian, chi2_pval_gaussian = chisquare(y_observed, expected_gaussian_vals)
print(f'chi2_value_gaussian = {chi2_value_gaussian}')
print(f'chi2_pval_gaussian = {chi2_pval_gaussian} \n\n')

yvals_gaussian = res_gauss.x[0] * func_gaussian_pdf(xvals, res_gauss.x[1], res_gauss.x[2])

ks_statistic = stats.ks_2samp(yvals_gaussian, y_observed)
print(f'ks_statistic = {ks_statistic}')

fig, ax = plotify.get_figax()
ax.hist(beta_init, bins=n_bins, range=(xmin, xmax), histtype='step', label="Particles", color=plotify.c_orange, linewidth=2)
ax.plot(xvals, yvals_gaussian)
plt.savefig(('plots/' + 'last_exercise_2'), facecolor=plotify.background_color, dpi=180)

plt.show()






n_bins = 50


xmin = np.min(theta)
xmax = np.max(theta)
theta_mean = np.mean(theta)
theta_std = np.std(theta)



expected_gaussian_vals = np.zeros(n_bins)
y_observed, xExp_edges = np.histogram(theta, bins=n_bins, range=(xmin, xmax))
xvals = np.linspace(xmin, xmax, n_bins)

x0 = [0.6, theta_mean, theta_std]
res_gauss = minimize(chi2_gauss, x0, args=(y_observed, xvals))
print(f'res_gauss gauss x = {res_gauss.x}')
print(f'res_gauss = {res_gauss}')

expected_gauss_vals = np.zeros(len(y_observed))
for i, x in enumerate(xvals):
	expected_gauss_vals[i] = res_gauss.x[0] * func_gaussian_pdf(x, res_gauss.x[1], res_gauss.x[2])

chi2_value_gaussian, chi2_pval_gaussian = chisquare(y_observed, expected_gaussian_vals)
print(f'chi2_value_gaussian = {chi2_value_gaussian}')
print(f'chi2_pval_gaussian = {chi2_pval_gaussian} \n\n')

yvals_gaussian = res_gauss.x[0] * func_gaussian_pdf(xvals, res_gauss.x[1], res_gauss.x[2])

fig2, ax2 = plotify.get_figax()

ax2.hist(theta, bins=n_bins, range=(xmin, xmax), histtype='step', label="Particles", color=plotify.c_orange, linewidth=2)
ax2.plot(xvals, yvals_gaussian)
plt.savefig(('plots/' + 'last_exercise'), facecolor=plotify.background_color, dpi=180)

print(f'theta_mean = {theta_mean}')
print(f'theta_std = {theta_std}')

plt.show()
