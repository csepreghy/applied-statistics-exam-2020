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
r = np.random

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

	return chi2

def get_weighted_mean(values, unvertainties):
	numerator = 0
	denominator = 0

	for value, uncertainty in zip(values, unvertainties):
		numerator += value / (uncertainty)
		denominator += 1 / (uncertainty)
	
	return numerator / denominator

def get_weighted_uncertainty(uncertainties):
	denominator = 0

	for uncertainty in uncertainties:
		denominator += 1 / (uncertainty)

	return np.sqrt(1 / denominator)

def chi2(values, uncertainties, n_parameters):
	mean = np.mean(values)
	weighted_mean = get_weighted_mean(values, uncertainties)

	std = np.std(values)
	ndof = len(values) - n_parameters
	chi2_value = 0

	if len(uncertainties.shape) == 0:
		uncertainties = [uncertainties] * len(values)

	for observed_value, uncertainty in zip(values, uncertainties):
		chi2_value += (observed_value - weighted_mean)**2 / uncertainty**2

	p_chi2 = stats.chi2.sf(chi2_value, ndof)

	return chi2_value, p_chi2

def exercise_1():
	n_parameters = 1
	hubble_values = np.array([73.5, 74, 73.3, 75, 67.6, 70.4, 67.66])
	hubble_uncertainties = np.array([1.4, 1.4, 1.8, 2, 0.7, 1.4, 0.42])
	weighted_mean = get_weighted_mean(hubble_values, hubble_uncertainties)
	weighted_uncertainty = get_weighted_uncertainty(hubble_uncertainties)

	print(f'weighted_mean = {weighted_mean}')
	print(f'weighted_uncertainty = {weighted_uncertainty}')

	chi2_value, p_chi2 = chi2(hubble_values, hubble_uncertainties, n_parameters)
	print(f'chi2_value = {chi2_value}')
	print(f'p_chi2 = {p_chi2}')

	# FIRST 4 MEASUREMENTS #

	hubble_values_4 = np.array([73.5, 74, 73.3, 75])
	hubble_uncertainties_4 = np.array([1.4, 1.4, 1.8, 2])

	chi2_value_4, p_chi2_4 = chi2(hubble_values_4, hubble_uncertainties_4, n_parameters)	
	print(f'chi2_value_4 = {chi2_value_4}')
	print(f'p_chi2_4 = {p_chi2_4}')

	# LAST 3 MEASUREMENTS #
	
	hubble_values_3 = np.array([67.6, 70.4, 67.66])
	hubble_uncertainties_3 = np.array([0.7, 1.4, 0.42])

	chi2_value_3, p_chi2_3 = chi2(hubble_values_3, hubble_uncertainties_3, n_parameters)	
	print(f'chi2_value_3 = {chi2_value_3}')
	print(f'p_chi2_3 = {p_chi2_3}')



	mean = np.mean(hubble_values)
	xvals = np.array([1, 2, 3, 4, 5, 6, 7])
	fig, ax = plotify.get_figax(figsize=(8,6), use_grid=True)

	plt.axhline(y=weighted_mean, linestyle='--', alpha=0.75, label='Weighted Mean', color=plotify.c_blue)
	plt.axhline(y=mean, linestyle='--', alpha=0.75, label='Mean', color=plotify.c_cyan)
	ax.scatter(xvals, hubble_values, c=plotify.c_orange)
	ax.errorbar(xvals, hubble_values, yerr=hubble_uncertainties, fmt='o', c=plotify.c_orange, capsize=2.5, label="Measurement with Uncertainty")
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.set_title("Hubble Constant Measurements")
	ax.set_xlabel("Measurement Number")
	ax.set_ylabel("Value / Uncertainty")
	plt.legend(facecolor="#282D33")
	
	# plt.savefig(('plots/' + 'hubble_constant'), facecolor=plotify.background_color, dpi=180)
	# plt.show()

def coulombs_law(F, d):
	k = 8.99e9
	Q = 10e-9
	
	q0 = (F * d**2) / (k * Q)
	return q0

def exercise_2():
	F = 0.87
	F_error = 0.08

	d = 0.0045
	d_error = 0.0003

	# SIMULATION #

	n_experiments = 10000
	F_vals = np.zeros(n_experiments)
	d_vals = np.zeros(n_experiments)
	q0s = []

	# for i in range(n_experiments):
	F_vals = r.normal(F, F_error, n_experiments)
	d_vals = r.normal(d, d_error, n_experiments)

	print(f'F = {F}')
	print(f'd = {d}')

	sigma = (d*d_error)**2 + (d*d_error)**2 + (d*F_error)**2 + (F*d_error)**2
	print(f'sigma = {sigma}')

	for F, d in zip(F_vals, d_vals):
		q0s.append(coulombs_law(F, d))

	q0 = np.mean(q0s)
	q0_error = np.std(q0s)
	print(f'q0 = {q0} ± ', q0_error)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def exercise_3():
	fDNA = np.loadtxt('data/data_DNAfraction.txt')
	print(f'len(fDNA) = {len(fDNA)}')
	n_bins = 25
	fDNA_min = np.min(fDNA)
	fDNA_max = np.max(fDNA)

	print(f'fDNA_max = {fDNA_max}')
	
	fDNA_mean = np.mean(fDNA)
	fDNA_std = np.std(fDNA)
	fDNA_rmse = rmse(fDNA, [fDNA_mean] * len(fDNA))

	print(f'fDNA_std = {fDNA_std}')
	print(f'fDNA_mean = {fDNA_mean}')
	print(f'fDNA_rmse = {fDNA_rmse}')

	# fig, ax = plotify.get_figax()
	# ax.hist(fDNA, bins=n_bins, range=(fDNA_min, fDNA_max), histtype='step', linewidth=2, color=plotify.c_orange)
	# ax.set_xlabel("Fraction of Neanderthal DNA")
	# ax.set_ylabel("Number of people in per bin")
	# ax.set_title("Histogram of people with different fractions of Neanderthal DNA")
	# plt.savefig(('plots/' + 'dna_distribution'), facecolor=plotify.background_color, dpi=180)

	# plt.show()

	xmin, xmax = 0.016, fDNA_max
	fDNA_main = []

	for i in fDNA:
		if i > 0.016: fDNA_main.append(i)
	
	fDNA_main_mean = np.mean(fDNA_main)
	fDNA_main_std = np.std(fDNA_main)

	expected_gaussian_vals = np.zeros(n_bins)
	y_observed, xExp_edges = np.histogram(fDNA_main, bins=n_bins, range=(xmin, xmax))
	xvals = np.linspace(xmin, xmax, n_bins)

	x0 = [0.6, fDNA_main_mean, fDNA_main_std]
	res_gauss = minimize(chi2_gauss, x0, args=(y_observed, xvals))
	print(f'res_gauss gauss x = {res_gauss.x}')
	print(f'res_gauss = {res_gauss}')

	expected_gauss_vals = np.zeros(len(y_observed))
	for i, x in enumerate(xvals):
		expected_gauss_vals[i] = res_gauss.x[0] * func_gaussian_pdf(x, res_gauss.x[1], res_gauss.x[2] - 0.4)
	
	chi2_value_gaussian, chi2_pval_gaussian = chisquare(y_observed, expected_gaussian_vals)
	print(f'chi2_value_gaussian = {chi2_value_gaussian}')
	print(f'chi2_pval_gaussian = {chi2_pval_gaussian} \n\n')

	print(f'yExp = {y_observed}')
	yvals_gaussian = res_gauss.x[0] * func_gaussian_pdf(xvals, res_gauss.x[1], res_gauss.x[2])

	fig, ax2 = plotify.get_figax()
	ax2.scatter(xvals, y_observed, s=3)
	# ax2.hist(fDNA_main, bins=n_bins, range=(xmin, xmax), histtype='step', linewidth=2, color=plotify.c_orange)
	ax2.plot(xvals, yvals_gaussian, color=plotify.c_blue)
	ax2.set_xlabel("Fraction of Neanderthal DNA (Main Population)")
	ax2.set_ylabel("Number of people in per bin")
	ax2.set_title("Histogram of people with different fractions of Neanderthal DNA")
	plt.show()

# params = [0.30825654 0.08781376]

if __name__ == "__main__":
	# exercise_1()
	# exercise_2()
	exercise_3()