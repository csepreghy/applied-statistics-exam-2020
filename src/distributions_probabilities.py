import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from iminuit import Minuit
from scipy import stats
from scipy.stats import binom, poisson, norm
from scipy.integrate import quad, simps
from scipy.optimize import minimize
import math

# plotify is a skin over matplotlib I wrote to automate styling of plots
# https://github.com/csepreghy/plotify
from plotify import Plotify

plotify = Plotify()
r = np.random

def func_binomial_pmf(x, n, p):
    return binom.pmf(np.floor(x+0.5), n, p)

def func_binomial_cdf(x, n, p):
    return binom.cdf(np.floor(x+0.5), n, p)


def func_poisson_pmf(x, lamb):
    return poisson.cdf(np.floor(x+0.5), lamb)
    # return poisson.pmf(np.floor(x+0.5), lamb)

def exercise_1():
	xmin = 0
	xmax = 144

	n = 144
	p = 1/2

	x = np.linspace(xmin, xmax, 1000)
	y = func_binomial_pmf(x, n, p)

	plotify.plot(x=x,
                 y=y,
                 title="Binomial PDF over the number of Wins",
                 xlabel="Number of Wins",
                 ylabel="P",
				 label="Binomial Distribution",
                 show_plot=False,
                 save=False)

	plt.axvline(x=72, linestyle='--', alpha=0.5, label='P of the score being exactly even')
	plt.legend(facecolor="#282D33")
	plt.savefig(('plots/' + 'football'), facecolor=plotify.background_color, dpi=180)
	plt.show()

	score_exactly_even = func_binomial_pmf(72, n, p)
	print(f'score_exactly_even = {score_exactly_even}')

def exercise_2():
	p = 0.054

	yvals = []
	xvals = np.linspace(0, 80, 80)
	major_yticks = np.linspace(0, 1, 11)

	for n in range(80):
		result = 1 - ((1 - p)**n)
		print(n, ": ", result)
		yvals.append(result)


	fig, ax = plotify.get_figax()
	for x, y in enumerate(yvals):
		plt.axvline(x=x, ymin=0, ymax=y, alpha=0.8, label='P of the score being exactly even', color=plotify.c_orange)
	
	# ax.scatter(xvals, yvals, color=plotify.c_orange, s=2)
	ax.set_yticks(major_yticks)
	ax.set_ylabel("P of hitting the window")
	ax.set_xlabel("Number of tries")
	ax.set_title("Probability of hitting the window")
	plt.savefig(('plots/' + 'window_hitting'), facecolor=plotify.background_color, dpi=180)
	plt.show()

if __name__ == "__main__":
	# exercise_1()
	exercise_2()