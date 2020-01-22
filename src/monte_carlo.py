import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import math

from scipy.integrate import quad, simps
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr, chisquare, norm
from scipy import stats

# plotify is a skin over matplotlib I wrote to automate styling of plots
# https://github.com/csepreghy/plotify
from plotify import Plotify

plotify = Plotify()
r = np.random

def pdf(x, C, c1, c2):
	return C * (c1 + x**c2)

def pdf_to_plot(x, C, xmin, xmax, n_bins, n_points, c1, c2):
	k = (xmax - xmin) / n_bins
	N = n_points * k
	return N * C * (c1 + x**c2)

def von_neumann(f, xmin, xmax, ymin, ymax, N_points, f_arg=()):
  x_accepted = []

  while len(x_accepted) < n_points:
    x = r.uniform(xmin, xmax)  
    y = r.uniform(0, ymax)

    if f(x, C, c1, c2) > y:
      x_accepted.append(x)
    
  return x_accepted

def pdf_to_minimize(params, accepted_x_vals):
	c1, c2 = params

	sampled_yvals, bin_edges = np.histogram(accepted_x_vals, bins=n_bins, range=(xmin, xmax))

	xvals = np.linspace(xmin, xmax, n_bins)
	expected_values = np.zeros(len(xvals))

	for i in range(len(xvals)):
		expected_values[i] = pdf_to_plot(xvals[i], C, xmin, xmax, n_bins, n_points, c1, c2)

	chi2_value, chi2_pval = chisquare(sampled_yvals, expected_values)

	return chi2_value

xmin = 1
xmax = 3
c1 = 1
c2 = 2

integral_C = quad(pdf, xmin, xmax, args=(1, c1, c2), epsabs=200)
print(f'integral_C = {integral_C}')

C = 1/integral_C[0]

integral_C = quad(pdf, xmin, xmax, args=(C, c1, c2), epsabs=200)
print(f'integral_C = {integral_C}')

xvals = np.linspace(xmin, xmax, 1000)
yvals = pdf(xvals, C, c1, c2)

# plotify.plot(xvals, yvals, show_plot=True)

pdf_mean = np.mean(yvals)
pdf_rms = np.std(yvals)

print(f'pdf_mean = {pdf_mean}')
print(f'pdf_rms = {pdf_rms}')

ymin = pdf(xmin, C, c1, c2)
ymax = pdf(xmax, C, c1, c2)

print(f'ymin = {ymin}')
print(f'ymax = {ymax}')

def fit_c_values(n_points, n_bins):
	# constants

	xmin = 1
	xmax = 3
	c1 = 1
	c2 = 2

	# Doing the rest

	accepted_x_vals = von_neumann(pdf, xmin, xmax, ymin, ymax, N_points=n_points, f_arg=C)

	xvals = np.linspace(xmin, xmax, n_points)
	yvals = np.zeros(len(xvals))

	for i in range(len(xvals)):
		yvals[i] = pdf_to_plot(xvals[i], C, xmin, xmax, n_bins, n_points, c1, c2)

	fig, ax = plotify.get_figax()

	ax.plot(xvals, yvals, c=plotify.c_blue, label="Probability Distribution")
	ax.hist(accepted_x_vals, bins=n_bins, range=(xmin, xmax), histtype='step', label="Number of Sampled Values", color=plotify.c_orange, linewidth=2)
	ax.set_xlabel("Randomly Sampled Value")
	ax.set_ylabel("Number Of Sampled Values")
	ax.set_title("Sampling Values According f(x)")
	ax.legend(facecolor="#282D33", loc="upper left")
	plt.savefig(('plots/' + 'monte_carlo'), facecolor=plotify.background_color, dpi=180)


	plt.show()
	
	pearson_correlation, _ = pearsonr(xvals, yvals)
	print('Pearsons correlation: %.3f' % pearson_correlation)

	x0 = [1, 2]
	res = minimize(pdf_to_minimize, x0, method='Nelder-Mead', tol=1e-4, args=accepted_x_vals)

	sampled_yvals, bin_edges = np.histogram(accepted_x_vals, bins=n_bins, range=(xmin, xmax))
	expected_values = np.zeros(len(sampled_yvals))
	xvals_bins = np.linspace(xmin, xmax, n_bins)
	for i in range(len(xvals_bins)):
		expected_values[i] = pdf_to_plot(xvals_bins[i], C, xmin, xmax, n_bins, n_points, c1, c2)

	chi2_value, chi2_pval = chisquare(sampled_yvals, expected_values)
	# print(f'chi2_value = {chi2_value}

	c1, c2 = res.x

	print(f'c1 = {c1}')
	print(f'c2 = {c2}')

	xvals = np.linspace(xmin, xmax, n_points)
	yvals = np.zeros(len(xvals))

	for i in range(len(xvals)):
		yvals[i] = pdf_to_plot(xvals[i], C, xmin, xmax, n_bins, n_points, c1, c2)


	fig, ax2 = plotify.get_figax()

	ax2.plot(xvals, yvals, c=plotify.c_blue, label="Fitted Probability Distribution")
	ax2.hist(accepted_x_vals, bins=n_bins, range=(xmin, xmax), histtype='step', label="Number of Sampled Values", color=plotify.c_orange, linewidth=2)
	ax2.set_xlabel("Randomly Sampled Value")
	ax2.set_ylabel("Number Of Sampled Values")
	ax2.set_title("Sampling Values According f(x) fitted")
	ax2.legend(facecolor="#282D33", loc="upper left")
	plt.savefig(('plots/' + 'monte_carlo_fitted'), facecolor=plotify.background_color, dpi=180)

	plt.show()

	c1_fraction = 1 - (1 / c1)
	c2_fraction = 1 - (2 / c2)

	return c1_fraction, c2_fraction

if __name__ == "__main__":
	n_points_list = [50000]
	n_bins = 75
	for n_points in n_points_list:
		c1_fraction, c2_fraction = fit_c_values(n_points, n_bins)
		print(f'c1_fraction = {c1_fraction}')
		print(f'c2_fraction = {c2_fraction}')


