# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import math
from scipy.stats import t

def regression_figure(x, y):
    plt.figure()
    plt.scatter(x, y)
    sxy = sum((x - x.mean())* (y - y.mean()))
    sxx = sum((x - x.mean())**2)
    syy = sum((y - y.mean())**2)
    ssr = (sxx*syy - sxy ** 2) / sxx
    # print sxy, sxx, syy, ssr
    b_estimator = sxy / sxx
    a_estimator = y.mean() - b_estimator * x.mean()
    # print b_estimator
    num = len(x)
    alpha = 0.1
    t_distr_value = t.isf(alpha / 2,num - 2)
    print 't value for confidence interval computation: ', t_distr_value
    interval_left_b = b_estimator - math.sqrt(ssr / ((num - 2) * sxx)) * t_distr_value
    interval_right_b = b_estimator + math.sqrt(ssr / ((num - 2) * sxx)) * t_distr_value
    interval_left_a = a_estimator - math.sqrt(sum(x ** 2) * ssr / (num * (num - 2) * sxx)) * t_distr_value
    interval_right_a = a_estimator + math.sqrt(sum(x ** 2) * ssr / (num * (num - 2) * sxx)) * t_distr_value
    plt.plot(x, a_estimator + b_estimator * x, linewidth = 7)
    plt.plot(x, interval_left_a + interval_left_b * x, linewidth = 1)
    plt.plot(x, interval_right_a + interval_right_b * x, linewidth = 1)
    plt.savefig('assign5_2_plot.pdf')
    
    print '------------------------------------------------'
    # judge hypothesis β = 0 at 1% level of significance
    test_stat = math.sqrt((num - 2) * sxx / ssr) * math.fabs(b_estimator)
    print 'value of test statistic: ', test_stat
    alpha_level = 0.01
    t_distr_value_judge = t.isf(alpha_level / 2,num - 2)
    print 't value for hypothesis test computation: ', t_distr_value_judge
    if test_stat > t_distr_value_judge:
        print 'hypothesis test result: reject'
    else:
        print 'hypothesis test result: accept'

if __name__ == '__main__':
#     The following data which relates x, the moisture of a wet mix of a certain product, 
#     to Y, the density of the finished product, is from the book: Introduction to Probability and Statistics for Engineers and Scientists by Sheldon M. Ross.
#     P361 example 9.3a

    x_i = Series([5,6,7,10,12,15,18,20])
    y_i = Series([7.4,9.3,10.6,15.4,18.1,22.2,24.1,24.8])
    regression_figure(x_i, y_i)
   
    

# <codecell>


# <codecell>


# <codecell>


