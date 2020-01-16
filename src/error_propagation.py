import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

from iminuit import Minuit
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chisquare, normaltest
import math

from plotify import Plotify

plotify = Plotify()
r = np.random

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

	n_experiments = 10
	F_vals = np.zeros(n_experiments)
	d_vals = np.zeros(n_experiments)
	q0s = []

	# for i in range(n_experiments):
	F_vals = r.normal(F, F_error, n_experiments)
	d_vals = r.normal(d, d_error, n_experiments)

	print(f'F = {F}')
	print(f'd = {d}')

	for F, d in zip(F_vals, d_vals):
		q0s.append(coulombs_law(F, d))

	print(f'q0s = {q0s}')



if __name__ == "__main__":
	# exercise_1()
	exercise_2()