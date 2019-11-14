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
import matplotlib.pyplot as plt
import seaborn as sns

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
        se_a = np.var(a, ddof = 1)/a.size
        se_b = np.var(b, ddof = 1)/b.size
        denom = np.sqrt(se_a + se_b)
        return np.abs(num/denom)
    
    def welch_df(a, b):
        S1 = np.var(a, ddof = 1)
        S2 = np.var(b, ddof = 1)
        N1 = a.size
        N2 = b.size
        V1 = N1 - 1
        V2 = N2 - 1
        num = (S1/N1 + S2/N2)**2
        denom = (S1/ N1)**2/V1 + (S2/ N2)**2/(V2)
        
    def p_value(a, b, two_sided = False):
        t = welch_t(a, b)
        df = welch_df(a, b)
        p = 1 - st.t.cdf(t, df)
        return p
    
       
def compare_pval_alpha(p_val, alpha):
    status = ''
    if p_val > alpha:
        status = "Fail to reject"
    else:
        status = 'Reject'
    return status  
    
class Two_Sample_Test():
    
    def visualize_dist(dist1,dist2):
        sns.set(color_codes=True)
        sns.set(rc={'figure.figsize':(12,10)})
        sns.distplot(dist1,hist=False, label="dist_1")
        sns.distplot(dist2,hist=False, label="dist_2")
        plt.legend()
        plt.show()
        
    def conclusion(result, t_crit, alpha):
        if (result[0]>t_crit) and (result[1]<alpha):
            print ("Null hypothesis rejected. H1 is accepted. Results are statistically significant with t-value =", round(result[0], 2), "critical t-value =", t_crit, "and p-value =", np.round((result[1]), 10))
        else:
            print ("Null hypothesis is True with t-value =", round(result[0], 2), ", critical t-value =", t_crit, "and p-value =", np.round((result[1]), 10))
    
    def visualize_t(t_stat, n_dist1, n_dist2):
        # initialize a matplotlib "figure"
        fig = plt.figure(figsize=(12,5))
        ax = fig.gca()
        # generate points on the x axis between -4 and 4:
        xs = np.linspace(-4, 4, 500)

        # use stats.t.ppf to get critical value. For alpha = 0.05 and two tailed test
        crit = stats.t.ppf(1-0.025, (n_dist1+n_dist2-2))

        # use stats.t.pdf to get values on the probability density function for the t-distribution

        ys= stats.t.pdf(xs, (n_dist1+n_dist2-2), 0, 1)
        ax.plot(xs, ys, linewidth=3, color='darkred')

        ax.axvline(t_stat, color='red', linestyle='--', lw=5,label='t-statistic')

        ax.axvline(crit, color='black', linestyle='--', lw=5)
        ax.axvline(-crit, color='black', linestyle='--', lw=5)

        plt.show()
        
    def t_crit(alpha, dof):
        return stats.t.ppf(1-alpha**2, dof)
    
    def t_test(dist1, dist2):
        return stats.ttest_ind(dist1, dist2)
    
    def visualize_sample_dist(dist):
        fig = plt.figure(figsize=(8,5))
        sns.distplot(dist)
        plt.show()
        print(stats.normaltest(dist))
        
    def explore_data(data1, data2):
        fig = plt.figure(figsize=(8,5))
        sns.boxplot( data1, data2,)
        plt.show()
        
    def overlapping_visual(data1,data2):
        fig = plt.figure(figsize=(8,5))
        sns.distplot(data1, hist=False)
        sns.distplot(data2, hist=False)
        plt.show()
        
    

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

def hypothesis_test_two():
    pass

def hypothesis_test_three():
    pass

def hypothesis_test_four():
    pass
