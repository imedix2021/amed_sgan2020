import csv
import pprint
import numpy as np
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

# benign(0) / malignant(1) csv data entry
# c_int: integer type array data　of correct answers 
# r_int: integer type array data　of prediction aswers using real images
# s_int: integer type array data　of prediction aswers using synthetic images
with open('./correctIncResNetV2_775_real.csv') as c:
    c_str = [str(n) for n in c]
    c_fl = [float(n) for n in c_str]
    c_int = [int(n) for n in c_fl]
    print(c_int)
with open('./predictionsIncResNetV2_775_real.csv') as r:
    r_str = [str(n) for n in r]
    r_fl = [float(n) for n in r_str]
    r_int = [int(n) for n in r_fl]
    print(r_int)

with open('./predictionsIncResNetV2_755_84000.csv') as s:
    s_str = [str(n) for n in s]
    s_fl = [float(n) for n in s_str]
    s_int = [int(n) for n in s_fl]
    print(s_int)

# array data preparation
c = np.array(c_int)
r = np.array(r_int)
s = np.array(s_int)

# correct / incorrect aggregatee
cr_compare = c == r
print (cr_compare)
cs_compare = c == s
print (cs_compare)
cm = confusion_matrix(cr_compare, cs_compare)
print (cm)

# define contingency table
#                             Synthetic
#                      Correct(1)   Incrrect(0)  
# Real    Correct(1)   Yes/Yes(YY)    Yes/No(FN)
#
print()
YY = cm[1][1]
NN = cm[0][0]
NY = cm[0][1]
YN = cm[1][0]
ar = np.array([[YY,YN],[NY,NN]])

# calculate mcnemar test
result = mcnemar(ar, exact=True)
# summarize the finding
print('statistic=%.3f, p-value=%.18f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')
