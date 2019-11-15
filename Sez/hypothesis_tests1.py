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


def hypothesis_test_one(alpha,df):
    short_players = df.loc[df['height'] < 200]
    tall_players = df.loc[df['height'] > 200]

    sh_pl_h = short_players.height
    sh_pl_3p = round(short_players['3P%'],3)

    tall_pl_h = tall_players.height
    tall_pl_3p = round(tall_players['3P%'],3)
    
    Two_Sample_Test.overlapping_visual(sh_pl_3p, tall_pl_3p)
    print("Since our dataset is non-normal, that means we'll need to use the Central Limit Theorem.")
    print("\n")
    print("esc + m + enter")
    print("\n")
    print("Now that we have helper functions to help us sample with replacement and calculate sample means, we just need to bring it all together and write a function that creates a sample distribution of sample means!")
    
    dist_size = 100
    short_dist = CLT.create_sample_distribution(sh_pl_3p, dist_size, n=30)
    tall_dist = CLT.create_sample_distribution(tall_pl_3p, dist_size, n=30)
    
    # Create a plot showing overlapping of distribution means and sds for inspection
    Two_Sample_Test.visualize_dist(short_dist, tall_dist)
    
    space = "\n\n"
    a = "1) Set up null and alternative hypotheses \n2) Choose a significance level \n3) Calculate the test statistic \n4) Determine the critical or p-value (find the rejection region) \n5) Compare t-value with critical t-value to reject or fail to reject the null hypothesis"
    b = "The Null Hypothesis"
    h0 = "𝐻0: The mean difference between short players' 3 point shooting percentage and tall players' 3 point shooting percentage is zero. i.e. 𝜇0=𝜇1"
    alter = "The Alternate Hypothesis"
    c = "In this example, the alternative hypothesis is that there is in fact a mean difference in 3 Points Shooting Percentage between short players and tall players."
    
    h1 = "𝐻1(2-tailed): The parameter of interest, our mean difference between short peoples' 3 point shooting percentage and tall players' 3 point shooting percentage, is different than zero."
    
    h1_1 = "𝐻1(1-tailed, >): TThe mean difference between short players' 3 point shooting percentage and tall players' 3 point shooting percentage is greater than zero."
    
    h1_1_1 = "𝐻1(1-tailed, <): The mean difference between short players' 3 point shooting percentage and tall players' 3 point shooting percentage is less than zero."
    
    print(a)
    print(space)
    print(b)
    print(space)
    print(h0)
    print(space)
    print(alter)
    print(space)
    print(h1)
    print(space)
    print(h1_1)
    print(space)
    print(h1_1_1)
    print(space)
    
    mean_diff = round(np.mean(short_dist) - np.mean(tall_dist), 3)
    n_short = len(short_dist)
    n_tall = len(tall_dist)
    dof = (n_short + n_tall - 2)
    t_crit = Two_Sample_Test.t_crit(alpha, dof)
    
    result  = Two_Sample_Test.t_test(short_dist, tall_dist)
    print(result)
    t_stat = result[0]
    p_value = result[1]
    
    Two_Sample_Test.visualize_t(t_stat, n_short, n_tall)
    
    Two_Sample_Test.conclusion(result, t_crit, alpha)

    pass

def hypothesis_test_two(alpha, df):
    skinny_players = df.loc[df['weight'] < 95]
    heavy_players = df.loc[df['weight'] > 95]

    sk_pl_w = skinny_players.weight
    sk_pl_fauls = round(skinny_players['PF/G'],3)

    heavy_pl_w = heavy_players.weight
    heavy_pl_fauls = round(heavy_players['PF/G'],3)
    
    #Taking sample distributions via Central Limit Theorem
    dist_size = 100
    skinny_dist =CLT.create_sample_distribution(sk_pl_fauls,dist_size, n=30)
    heavy_dist = CLT.create_sample_distribution(heavy_pl_fauls,dist_size, n=30)
    
    # Create a plot showing overlapping of distribution means and sds for inspection
    Two_Sample_Test.visualize_dist(skinny_dist, heavy_dist)
    
    a = "1) Set up null and alternative hypotheses \n2) Choose a significance level \n3) Calculate the test statistic \n4) Determine the critical or p-value (find the rejection region) \n5) Compare t-value with critical t-value to reject or fail to reject the null hypothesis"
    b = "The Null Hypothesis"
    null_hyp = "𝐻0 H0: On average, heavy players' fauls percentage equals to the skinny players' faul percentage. i.e. 𝜇0=𝜇1"
    h1 = "𝐻1(2-tailed): The parameter of interest, our mean difference between heavy players' fauls percentage and skinny players' fauls percentage, is different than zero."
    h1_1 = "𝐻1(1-tailed, >): The mean difference between heavy players' fauls percentage and skinny players' fauls percentage is greater than zero. "
    h1_1_1 = "𝐻1(1-tailed, <): The mean difference between heavy players' fauls percentage and skinny players' fauls percentage is less than zero."
    space = "\n\n"
    print(a)
    print(space)
    print(b)
    print(space)
    print(null_hyp)
    print(space)
    print(h1)
    print(space)
    print(h1_1)
    print(space)
    print(h1_1_1)
    print(space)
    
    mean_diff = round(np.mean(heavy_dist) - np.mean(skinny_dist), 3)
    n_heavy = len(heavy_dist)
    n_skinny = len(skinny_dist)
    dof = (n_heavy + n_skinny - 2)
    
    t_crit = Two_Sample_Test.t_crit(alpha, dof)
    
    result  = Two_Sample_Test.t_test(heavy_dist, skinny_dist)
    print(result)
    t_stat = result[0]
    p_value = result[1]
    
    Two_Sample_Test.visualize_t(t_stat, n_heavy, n_skinny)
    
    Two_Sample_Test.conclusion(result, t_crit, alpha)
    
    
pass

def hypothesis_test_three():
    pass

def hypothesis_test_four():
    pass
