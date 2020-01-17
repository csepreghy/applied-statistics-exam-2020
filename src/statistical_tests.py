import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from iminuit import Minuit
from scipy import stats
from scipy.stats import binom, poisson, norm
from scipy.integrate import quad, simps
from scipy.optimize import minimize
import math

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC

# plotify is a skin over matplotlib I wrote to automate styling of plots
# https://github.com/csepreghy/plotify
from plotify import Plotify

plotify = Plotify()

# This script reads the file "data_Cells.txt" for problem 4.1 in the exam of 
# Applied Statistics 2019/20. This data files contains 4690 entries in three
# column, where the three numbers are the cell label (0 for P cell, 1 for E 
# cell), the cell size (in micro meters), and the cell transparency. The script 
# was distributed along with the exam and the data file itself on the 15th of 
# January 2020.

celltype_list, size_list, transp_list = np.loadtxt('data/data_Cells.txt', unpack=True)

P_less_than_9 = []
E_more_than_9 = []

for celltype, size in zip(celltype_list, size_list):
	if size < 9: P_less_than_9.append([celltype, size])
	else: E_more_than_9.append([celltype, size])


type1_error = 0
type2_error = 0

for i in P_less_than_9:
	if i[0] == 1.0: type2_error += 1

type2_ratio = type2_error / len(P_less_than_9)
print(f'type2_error = {type2_error}')
print(f'type2_ratio = {type2_ratio}')

for i in E_more_than_9:
	if i[0] == 0.0: type1_error += 1

type1_ratio = type1_error / len(E_more_than_9)
print(f'type1_error = {type1_error}')
print(f'type1_ratio = {type1_ratio}')

X = np.column_stack((size_list, transp_list))
print(f'X = {X}')
y = celltype_list

# --------------------------------------------------------- #
# -------------- Which variable is better ----------------- #
# --------------------------------------------------------- #

clf_svm_size = SVC(gamma='auto')
X_size = size_list.reshape(-1, 1) 
X_transp = transp_list.reshape(-1, 1)
y = celltype_list

clf_svm_size.fit(X_size, y)
print('size score: ', clf_svm_size.score(X_size, y))

clf_svm_size.fit(X_transp, y)
print('transp score: ', clf_svm_size.score(X_transp, y))


# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

print(f'y_train = {y_train}')

# Learn to predict each class against the other
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

ROC_aur_stuff = roc_auc_score(y_test, y_pred)


fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plotify.plot(fpr, tpr)


