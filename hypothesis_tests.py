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
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm

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
        p = 1 - st.t.cdf(t, df)
        return p

def cohens_d(a, b):
    return np.abs((np.mean(a)-np.mean(b))/np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2.0))
    
class Hypothesis_Testing():
    
    def compare_pval_alpha(p_val, alpha):
        status = ''
        if p_val > alpha:
            status = "P-value is greater than alpha. Fail to reject null hypothesis"
        else:
            status = 'P-value is less than alpha. Reject the null hypothesis and accept the alternative'
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


class HT3():
    
    def value_counts(data):
        display(data.POS.value_counts())
        print("Our data shows that we have {} Point Guards and {} Shooting Guards".format(358, 349))
        print("There also seem to be players that play various positions. We will drop these from our dataframe to avoid sampling them")   
        
    def make_PG_df(data):
        PG = data.loc[(data.POS == 'PG')]
        PG_FT = PG['FT%']
        return PG_FT
    
    def make_SG_df(data):
        SG = data.loc[(data.POS == 'SG')]
        SG_FT = SG['FT%']
        return SG_FT
    
    def full_hypothesis_test(alpha, data):
        np.random.seed(100)
        step1 = HT3.value_counts(data)
        PG_subset = HT3.make_PG_df(data)
        SG_subset = HT3.make_SG_df(data)
        
        PG_sample = CLT.create_sample_distribution(PG_subset, dist_size = 150, n = 50)
        SG_sample = CLT.create_sample_distribution(SG_subset, dist_size = 150, n = 50)
        
        PG_test = st.normaltest(PG_sample)

        print("\nPoint guards sample characteristics:")
        print("\nSample Distribution mean: {}".format(np.mean(PG_sample)))
        print("Sample Distribution median: {}".format(np.median(PG_sample)))
        print("Sample Distribution variance: {}".format(np.var(PG_sample)))
        print("Sample Distribution standard deviation: {}".format(np.std(PG_sample)))


        print("\nResults for normality of point guards sample: \nStatistic: {} \np-value: {}" .format(PG_test[0], PG_test[1]))
        if PG_test[1] > 0.05:
            print("Cannot reject the null hypothesis that the distribution is normal\n")

        SG_test = st.normaltest(SG_sample)
    
        print("\nShooting guards sample characteristics:")
        print("\nSample Distribution mean: {}".format(np.mean(SG_sample)))
        print("Sample Distribution median: {}".format(np.median(SG_sample)))
        print("Sample Distribution variance: {}".format(np.var(SG_sample)))
        print("Sample Distribution standard deviation: {}".format(np.std(SG_sample)))


        print("\nResults for normality of shooting guards sample: \nStatistic: {} \np-value: {}" .format(SG_test[0], SG_test[1]))
        if SG_test[1] > 0.05:
            print("Cannot reject the null hypothesis that the distribution is normal\n")
        
        plt.figure(figsize = (15,8))
        sns.distplot(PG_sample, hist = True, hist_kws = {
                                                "linewidth": 2,
                                                "edgecolor": 'blue',
                                                "alpha": 0.5,
                                                "color": "w",
                                                "label": 'Point Guards'})
        sns.distplot(SG_sample, hist = True, hist_kws = {
                                                "linewidth": 2,
                                                "edgecolor": 'red',
                                                "alpha": 0.5,
                                                "color": "w",
                                                "label": "Shooting Guards"})
        plt.title("Comparing Distributions")
        plt.legend()
        
        print("\nThere seems to be a distinct difference in the distribution of sample means between point guards and shooting guards. We can test for difference by using Welch's t-test")
        
        p, t = Welchs_Test.p_value(PG_sample, SG_sample, two_sided = False), Welchs_Test.welch_t(PG_sample, SG_sample)
        print("Sample t-statistic: {} \nSample p-value: {}".format(t, p))
              
        alpha = 0.05
        t_crit = st.t.ppf(0.95, df = Welchs_Test.welch_df(PG_sample, SG_sample))
        
        if t > t_crit:
              print("Our Welch's t-statistic of {} is greater than our t critical value of {}\nthus we can reject our null hypothesis that there is no difference in free throw percentage between \npoint guards and shooting guards".format(t, t_crit))
        
        print("\nUsing Cohen's d coefficient we can observe the effect size between the two means of our sample distributions")
        print("The Cohen's d coefficient between the PG sample and the SG sample is {}".format(cohens_d(PG_sample, SG_sample)))
        cd = cohens_d(PG_sample, SG_sample)
        if cd >= 0.80:
            print("A Cohen's D of greater than 0.80 is considered to be a large effect size")

class HT4():
    
    def make_subsets(data):
        PG_subset = data.loc[data['POS'] == 'PG']
        SG_subset = data.loc[data['POS'] == 'SG']
        SF_subset = data.loc[data['POS'] == 'SF']
        PF_subset = data.loc[data['POS'] == 'PF']
        C_subset = data.loc[data['POS'] == 'C']
        list_of_data_frames = [PG_subset, SG_subset, SF_subset, SF_subset, C_subset]
        combined = pd.concat(list_of_data_frames)
        return combined
        
    def make_column_names(data):
        column_names = ['POS','BLK/G']
        new_df = data.loc[:, column_names]
        return new_df
    
    def reset_index(data):
        data.reset_index(inplace = True)
        data.drop('index', axis = 1, inplace = True)
        return data
    
    def rename_column(data):
        data.rename(columns = {'BLK/G':'BLK'}, inplace = True)
        return data
    
    def full_hypothesis_test(data):
        
        step1 = HT4.make_subsets(data)
        step2 = HT4.make_column_names(step1)
        step3 = HT4.reset_index(step2)
        final = HT4.rename_column(step3)
        
        formula = "BLK ~ C(POS)"
        lm = ols(formula, final).fit()
        table = sm.stats.anova_lm(lm, typ = 2)
        print("Below is our table of results for an ANOVA test\n")
        print(table)
        
        print("\nOur f-critical value of 301 is far greater than the critical value of 4.2 at degrees of freedom 4 and alpha level 0.05 \nThus we can reject the null hypothesis that there is no difference in the means of blocks per game across positions\n")
        
        print("\nPairwise comparisons may be more meaninful due to the range of results we have exhibited")
        
        SG_subset = final.loc[final['POS'] == 'SG']
        SG_blocks = SG_subset['BLK']
        
        SG_BLK_sample_distribution = CLT.create_sample_distribution(SG_blocks, dist_size = 100, n = 30)
        print("\nBelow we can see the shooting guards sample means distribution plot of blocks per game")
        plt.figure(figsize = (15,8))
        sns.distplot(SG_BLK_sample_distribution, hist = True, hist_kws = {
                                                "linewidth": 2,
                                                "edgecolor": 'red',
                                                "alpha": 0.5,
                                                "color": "w",
                                                "label": "Shooting Guards Sample Distribution"})
        plt.xlabel("Average Blocks per Game")
        plt.ylabel("Distplot")
        plt.title("Shooting Guards Blocks per Game")
        plt.legend()

        
        
        PG_subset = final.loc[final['POS'] == 'PG']
        PG_blocks = PG_subset['BLK']
       
        PG_BLK_sample_distribution = CLT.create_sample_distribution(PG_blocks, dist_size = 100, n = 30)
        print("\nBelow we can see the point guards sample means distribution plot of blocks per game")
        plt.figure(figsize = (15,8))
        sns.distplot(PG_BLK_sample_distribution, hist = True, hist_kws = {
                                                "linewidth": 2,
                                                "edgecolor": 'red',
                                                "alpha": 0.5,
                                                "color": "w",
                                                "label": "Point Guards Sample Distribution"})
        plt.xlabel("Average Blocks per Game")
        plt.ylabel("Distplot")
        plt.title("Point Guards Blocks per Game")
        plt.legend()
        
        
        print("\nThere seems to be a distinct difference in the distribution of sample means between point guards and shooting guards. We can test for difference by using Welch's t-test")
        
        p, t = Welchs_Test.p_value(PG_BLK_sample_distribution, SG_BLK_sample_distribution, two_sided = False), Welchs_Test.welch_t(PG_BLK_sample_distribution, SG_BLK_sample_distribution)
        print("Sample t-statistic: {} \nSample p-value: {}".format(t, p))
              
        alpha = 0.05
        t_crit = st.t.ppf(0.95, df = Welchs_Test.welch_df(PG_BLK_sample_distribution, SG_BLK_sample_distribution))
        
        if t > t_crit:
              print("Our Welch's t-statistic of {} is greater than our t critical value of {}\nthus we can reject our null hypothesis that there is no difference in blocks made per game between point guards and shooting guards".format(t, t_crit))
        
        print("\nUsing Cohen's d coefficient we can observe the effect size between the two means of our sample distributions")
        print("The Cohen's d coefficient between the PG sample and the SG sample is {}".format(cohens_d(PG_BLK_sample_distribution, SG_BLK_sample_distribution)))
        cd = cohens_d(PG_BLK_sample_distribution, SG_BLK_sample_distribution)
        if cd >= 0.80:
            print("A Cohen's D of greater than 0.80 is considered to be a large effect size")
        elif cd < 0.80:
            print("A Cohen's D of this value is considered to be a medium effect")
        elif cd < 0.5:
            print("A Cohen's D of this value is considered to be a small effect")
        
        a = Hypothesis_Testing.compare_pval_alpha(p_val = p, alpha = 0.05)
        print(a)
        
