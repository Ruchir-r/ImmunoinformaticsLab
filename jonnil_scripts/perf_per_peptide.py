import pandas as pd
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binomtest, ttest_rel
import numpy as np
from itertools import combinations
import os
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # suppress UserWarnings


# methods = ['blosum50','blosum_sw','tcrlang-no-normalization','tcrlang-linear']
methods = ['tcrlang-baseline', 'tcrlang-baseline-norm','tcrlang-sampleweight-norm','tcrlang-swn-lr','tcrlang-swn-pruned','tcrlang-swn-tanh']

names = methods

perf_dicts = []

metrics = ['AUC','AUC 0.1']

peptides_set = set()
pep_subset = set(['SPRWYFYYL', 'GILGFVFTL', 'KLGGALQAK', 'CINGVCWTV', 'RAKFKQLL', 'AVFDRKSDAK', 'RFPLTFGWCF', 'KSKRTPMGF', 'ELAGIGILTV', 'NLVPMVATV', 'ATDALMTGF', 'GLCTLVAML', 'SLFNTVATLY', 'LLWNGPMAV', 'DATYQRTRALVR', 'IVTDFSVIK', 'FEDLRLLSF', 'HPVTKYIM', 'GPRLGVRAT', 'VLFGLGFAI', 'YLQPRTFLL', 'FEDLRVLSF', 'RLPGVLPRA', 'RLRAEAQVK', 'RPPIFIRRL', 'CTELKLSDY'])

N = 50

for i,m in enumerate(methods):
    #df = pd.read_csv(f'{m}/output/kfold_pred_experiment.csv') # Change this path to your model's cross-validated prediction CSV file

    df = pd.read_csv(f'model_results/{m}/kfold_pred_experiment.csv') # Change this path to your model's cross-validated prediction CSV file

    perf = {p:{} for p in metrics}
    
    # if i==1:
    counts = df[df['binder'] > 0]['peptide'].value_counts().to_dict()

    for pep in pep_subset:

        pep_df = df[df['peptide'] == pep]

        n = len(pep_df[(pep_df['binder'] > 0)])

        peptides_set.add(pep)

        try:
            perf['AUC'][pep] = roc_auc_score(pep_df['binder'], pep_df['prediction'])
            perf['AUC 0.1'][pep] = roc_auc_score(pep_df['binder'], pep_df['prediction'], max_fpr=0.1)
        except ValueError:
            perf['AUC'][pep] = 0
            perf['AUC 0.1'][pep] = 0

    perf_dicts.append(perf)


peptides = sorted(list(peptides_set), key=lambda p: counts[p], reverse=True)

metrics = ['AUC','AUC 0.1']

fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(8,4))
fig2, axes2 = plt.subplots(nrows=len(metrics), ncols=1, figsize=(20,8))

for i, metric in enumerate(metrics):

    d = {}

    for k, m in enumerate(names):
        d[m] = {p:perf_dicts[k][metric][p] for p in peptides}

    p_values = {}
    for x, y in list(combinations(range(len(methods)), 2)):

        p = ttest_rel([perf_dicts[x][metric][p] for p in peptides],[perf_dicts[y][metric][p] for p in peptides], alternative='two-sided').pvalue
        if p < 0.05:
            p_values[(x,y)] = p

    medians = [np.median([perf_dicts[k][metric][p] for p in peptides]) for k in range(len(methods))]

    # Bar plot
    data = pd.DataFrame(d)
    data['count'] = data.index.map(counts)
    data = data.sort_values('count', ascending=False).drop('count',axis=1)

    ax = axes2[i]
    data.plot.bar(ax=ax, width=0.75, edgecolor='black', legend=False,)

    ax.set_ylabel(metric, fontsize=12)
    ax.axhline(y=0.5, color='red', linestyle='dashed', linewidth=0.5, zorder=-99)
    ax.axhline(y=0.55, color='blue', linestyle='dashed', linewidth=0.5, zorder=-99)

    labels = [pep + f" - {counts[pep]}" for pep in data.index]
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)

    # Box plot
    medians = [np.median([perf_dicts[k][metric][p] for p in peptides]) for k in range(len(methods))]
    
    ax = axes[i]
    data.replace(0, np.nan, inplace=True)
    box_plot = sns.boxplot(data, color='white', ax=ax, showfliers=False)
    sns.stripplot(data, ax=ax, size=2.5)

    ax.set_yticks([tick for tick in ax.get_yticks() if tick <= 1.0])

    ax.set_ylabel(metric, fontsize=12)

    vals = list(d[names[0]].values())
    median = np.median(vals)
    low, high = np.quantile(vals, 0.25), np.quantile(vals, 0.75)
    ax.axhline(y=median, color='red', linestyle='dashed', linewidth=0.5, zorder=-99)
    ax.axhline(y=low, color='red', linestyle='dashed', linewidth=0.5, zorder=-99)
    ax.axhline(y=high, color='red', linestyle='dashed', linewidth=0.5, zorder=-99)
    ax.set_xticklabels(data.columns, fontsize=10, rotation=30, ha='right')

    for xtick in box_plot.get_xticks():
        print(f"{xtick} {data.iloc[:, xtick].median(skipna=True)}")
        box_plot.text(xtick,data.iloc[:, xtick].median(skipna=True) ,round(data.iloc[:, xtick].median(skipna=True) ,3), fontsize=6,
                        horizontalalignment='center',color='black',weight='semibold')

handles = [Patch(facecolor=sns.color_palette()[i], edgecolor='black') for i in range(len(names))]

labels = names
fig2.legend(handles, labels, loc='lower center', ncols=len(labels),)
fig2.tight_layout()
fig2.subplots_adjust(bottom=0.2) # or whatever


fig.tight_layout()

fig.savefig('figures/box_tcrlang_models.pdf')
fig2.savefig('figures/bar_tcrlang_models.pdf')

