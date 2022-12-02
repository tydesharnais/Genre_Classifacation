# import pandas and numpy
import pandas as pd
import numpy as np
# statistical analysis imports
from math import sqrt
from scipy import stats
# viz imports
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib as mpl
import seaborn as sns
from cycler import cycler
# default viz size settings
sns.set(rc={'figure.figsize':(14, 10)})
sns.set_context("talk", rc={"font.size":14,"axes.titlesize":18,"axes.labelsize":14}) 
plt.rc('figure', figsize=(14, 10))
plt.rc('font', size=12)
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.figsize'] = 14, 10
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linestyle'] = '--'
mpl.rcParams['axes.prop_cycle'] = cycler(color=['deepskyblue', 'firebrick', 'darkseagreen', 'violet'])


def catplot_viz(df, x_Param : str, y_Param : str, plt_Kind : str):
    '''
        Categorical scatterplots:

    stripplot() (with kind="strip"; the default)

    swarmplot() (with kind="swarm")

    Categorical distribution plots:

    boxplot() (with kind="box")

    violinplot() (with kind="violin")

    boxenplot() (with kind="boxen")

    Categorical estimate plots:

    pointplot() (with kind="point")

    barplot() (with kind="bar")

    countplot() (with kind="count")
        This function produces a swarm plot on explicit tracks' and non-explicit tracks' popularity.
        plt_kind ex boxen, swarm, bar, strip
    '''

    #print('Does a track being explicit or not correlate with its popularity?')
    sns.catplot(x=x_Param, y=y_Param, kind=plt_Kind, data=df, height=8, aspect=1)
    plt.title(label=f'Does a tracks {x_Param} or not correlate with its {y_Param}?')
    plt.show()

def feature_ttest(df,x_Param : str, y_Param : str ,alpha=0.05):
    '''
    This function takes in a DataFrame and an alpha value (default is .05)
    and prints off the Independent T-Test to compare mean popularity
    of explicit tracks versus non-explicit tracks.

    X_Param : what you are testing
    Y_Param : what you are testing against 
    '''
    print('Set the alpha/significance level:')
    print('  alpha =', alpha)
    
    print('\n---\n')
    
    print('Check for normal distribution:')
    sns.distplot(df[x_Param])
    plt.show()
    
    print('---\n')
    df[f'{x_Param}_bins'] = pd.qcut(x=df[x_Param], q=2, labels=['low','high']) # cut into high/low bins for variance
    print('Check values counts:')
    print(df[y_Param].value_counts())
    
    print('\n---\n')

    
    print('Compare variances:')
    explicit_sample = df[df[f'{x_Param}_bins']=='low'].popularity #lower x param (via dance, explit, etc)
    not_explicit_sample = df[df[f'{x_Param}_bins']=='high'].popularity
    
    # if [results of lavenes variance test], then equal_var = __ (automate checking similar variance)
    print(explicit_sample.var())
    print(not_explicit_sample.var())
          
    print("They are of relatively equal variance, so we will set the argument of equal_var to True. After the MVP this will be done with the Levene test instead of by hand.")
    
    print('\n---\n')
          
    print("Compute test statistic and probability (t-statistic & p-value)")
    t, p = stats.ttest_ind(explicit_sample, not_explicit_sample, equal_var = True)
    print('Test statistic:', t, '\np-value:', p/2, '\nalpha:', alpha)
    
    print('\n---\n')
    
    null_hypothesis = "there is no significant difference between the mean popularity of explicit tracks and non-explicit tracks."
    if p/2 < alpha:
        print("We reject the hypothesis that", null_hypothesis)
    else:
        print("We fail to reject the null hypothesis.")
        
    print('\n---\n')
          
    print('mean of non-explicit songs:', not_explicit_sample.mean(), '\nmean of explicit songs:', explicit_sample.mean())

def param_viz(train, x_Param, y_Param):
    '''
    Produces visualizations that answer the question:
    Is there a difference in mean popularity across dancebility bins?
    '''
    # First Viz
    # visualizing each observation by release date and popularity
    plt.figure(figsize=(12,6))

    sns.scatterplot(x=train[x_Param], y=train[y_Param])
    # reference line for overall y_Param average
    plt.axhline(train[y_Param].mean(),linestyle='-',label=f'Train {y_Param} Average', color='black')
    plt.axvline(train[x_Param].mean(), linestyle='--',label=f'Train {x_Param} Average', color='black')

    plt.title(f'{x_Param} vs. {y_Param}', size=15)

    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # line break
    print("\n")

    # Second Viz
    # bin danceability for better visualizing
    train[f'{x_Param}_bins'] = pd.qcut(x=train[x_Param], q=3, labels=['low','medium','high'])

    # to plot reference line of overall train average y_Param
    y_Param_rate = train[y_Param].mean()

    plt.figure(figsize=(12,6))

    # plots the average of each features subgroups as bar plots
    sns.barplot(f'{y_Param}', f'{x_Param}_bins', data=train, alpha=.8)
    plt.xlabel('')
    plt.ylabel(f'{x_Param} Bins', size=13)
    plt.title(f'{y_Param} Rate by {x_Param}', size=16)
    plt.axvline(y_Param_rate, ls='--', color='grey', label='Overall Average')

    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.tight_layout()
    plt.show()

def corr_heatmap(train, param_Corr : str):

    # put popularity in first position
    heatmap_data = train
    first_col = heatmap_data.pop(param_Corr)
    heatmap_data.insert(0, param_Corr, first_col)

    # create correlation heatmap
    corr = heatmap_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    ax = sns.heatmap(corr, mask=mask, center=0, cmap=sns.diverging_palette(95, 220, n=250, s=93, l=35), square=True) 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
    plt.title('Which features have significant linear correlation?')
    ax

