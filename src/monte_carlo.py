import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import math

from scipy.integrate import quad, simps
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr, chisquare, norm

from distributions_probability import func_gaussian_pdf
from plotify import Plotify

plotify = Plotify()
r = np.random