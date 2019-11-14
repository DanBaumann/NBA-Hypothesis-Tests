"""
This module is for your final hypothesis tests.
Each hypothesis test should tie to a specific analysis question.

Each test should print out the results in a legible sentence
return either "Reject the null hypothesis" or "Fail to reject the null hypothesis" depending on the specified alpha
"""

import pandas as pd
import numpy as np
from scipy import stats
import math
import scipy.stats as st

class CLT():

    def get_sample(data, n):
        sample = []
        while len(sample) != 30:
            x = np.random.choice(round(data, 3))
            sample.append(x)
        return sample
    
    def get_sample_mean(sample):
        return np.sum(sample)/len(sample)
    
    def create_sample_distribution(data, dist_size, n = 30):
        sample_dist = []
        while len(sample_dist) != dist_size:
            sample = CLT.get_sample(data, n)
            sample_mean = CLT.get_sample_mean(sample)
            sample_dist.append(sample_mean)
        return sample_dist
    
class Welchs_Test():
    
    def welch_t(a, b): 
        num = np.mean(a) - np.mean(b)
        se_a = np.var(a, ddof = 1)/len(a)
        se_b = np.var(b, ddof = 1)/len(b)
        denom = np.sqrt(se_a + se_b)
        return np.abs(num/denom)
    
    def welch_df(a, b):
        S1 = np.var(a, ddof = 1)
        S2 = np.var(b, ddof = 1)
        N1 = len(a)
        N2 = len(b)
        V1 = N1 - 1
        V2 = N2 - 1
        num = (S1/N1 + S2/N2)**2
        denom = (S1/ N1)**2/V1 + (S2/ N2)**2/(V2)
        return num/denom
        
    def p_value(a, b, two_sided = False):
        t = Welchs_Test.welch_t(a, b)
        df = Welchs_Test.welch_df(a, b)
        p = 1 - (st.t.cdf(t, df))
        return p
    
class Normal_Test():
    
    def normality_test(name, data):
        test = st.normaltest(data)
        print("Results for {} data: \nt-statistic: {} \np-value: {}".format(name, test[0], test[1]))
        if test[1] < 0.05:
            print("Can reject the null hypothesis that this distribution is normal")
        else:
            print("Cannot reject the null hypothesis that this distribution is normal")

    
class hypothesis_testing():
    
    def compare_pval_alpha(p_val, alpha):
        status = ''
        if p_val > alpha:
            status = "Fail to reject the null hypothesis"
        else:
            status = 'Reject the null hypothesis. Accept the alternative hypothesis'
        return status   
    

# def hypothesis_test_one(alpha = None, cleaned_data):
#     """
#     Describe the purpose of your hypothesis test in the docstring
#     These functions should be able to test different levels of alpha for the hypothesis test.
#     If a value of alpha is entered that is outside of the acceptable range, an error should be raised.

#     :param alpha: the critical value of choice
#     :param cleaned_data:
#     :return:
#     """
#     # Get data for tests
#     comparison_groups = create_sample_dists(cleaned_data=None, y_var=None, categories=[])

#     ###
#     # Main chunk of code using t-tests or z-tests, effect size, power, etc
#     ###

#     # starter code for return statement and printed results
#     status = compare_pval_alpha(p_val, alpha)
#     assertion = ''
#     if status == 'Fail to reject':
#         assertion = 'cannot'
#     else:
#         assertion = "can"
#         # calculations for effect size, power, etc here as well

#     print(f'Based on the p value of {p_val} and our aplha of {alpha} we {status.lower()}  the null hypothesis.'
#           f'\n Due to these results, we  {assertion} state that there is a difference between NONE')

#     if assertion == 'can':
#         print(f"with an effect size, cohen's d, of {str(coh_d)} and power of {power}.")
#     else:
#         print(".")

#     return status
