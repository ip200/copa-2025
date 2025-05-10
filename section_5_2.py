import numpy as np
import pandas as pd


from utils.section_5_2.experiment_desc import run_exp

OUTPUT_DIR = './results/section_5_2/'

classifiers = ['rf', 'xb', 'lr']

cal_names = [
    'base',
    'platt_const',
    'platt_windowed',
    'platt_continuous',
    'platt_continuous_beat',
    'platt_continuous_calibeat',
    'protected',
    'va_protected',
    'protected_va',
    'protected_beat',
    'platt_va']

seeds = range(100, 110)
datasets = {
    'churn': [0, 1, 2, 3],
    'bank': [0, 1, 2],
    'credit': [0, 1, 2],
    'fetal': [0, 1],
}

for dataset, design_list in datasets.items():
    output_ece = pd.DataFrame()
    output_sharp = pd.DataFrame()
    for classifier in classifiers:
        for design in design_list:
            for seed in seeds:
                print(classifier, dataset, design, seed)
                test = run_exp(dataset, seed=seed, design=design, classifier=classifier)
                result_list = np.zeros((len(cal_names), len(test[0])))
                for i in range(len(cal_names)):
                    result_list[i, :] = test[i]
                scratch = pd.DataFrame(result_list)
                scratch['calibrator'] = cal_names
                scratch['dataset'] = dataset
                scratch['seed'] = seed
                scratch['design'] = design
                scratch['classifier'] = classifier
                output_ece = pd.concat((output_ece, scratch))
                output_ece.to_csv(OUTPUT_DIR + 'output_ece_binning_' + dataset + '.csv')

                result_list = np.zeros((len(cal_names), len(test[0])))
                for i in range(len(cal_names)):
                    result_list[i, :] = test[i + len(cal_names)]
                scratch = pd.DataFrame(result_list)
                scratch['calibrator'] = cal_names
                scratch['dataset'] = dataset
                scratch['seed'] = seed
                scratch['design'] = design
                scratch['classifier'] = classifier
                output_sharp = pd.concat((output_sharp, scratch))
                output_sharp.to_csv(OUTPUT_DIR + 'output_sharp_binning_' + dataset + '.csv')


#  ------ write to tex ----------



mapper = {
    'base' : 'Base model (BM)',
    'platt_const' : 'Fixed-batch Platt scaling (FPS)',
    'platt_windowed': 'Windowed Platt scaling (WPS)',
    'platt_continuous': 'Online Platt scaling (OPS)',
    'platt_continuous_beat': 'OPS + tracking (TOPS)',
    'platt_continuous_calibeat': 'OPS + hedging (HOPS)',
    'protected':'BM - protected',
    'protected_beat':'HOPS - protected',
    'va_protected': 'Venn-abers  - protected'}


def parse_results(dset, cf, ds, typ='ece'):
    if typ=='ece:':
        results = pd.read_csv(OUTPUT_DIR + 'output_ece_binning_' + dset + '.csv')
        results.drop(['dataset'], axis=1, inplace=True)
        opa = results[results.classifier==cf].copy()
        opa.drop(['classifier'], axis=1, inplace=True)
        opa = opa.groupby(['calibrator', 'design']).mean().drop(['Unnamed: 0', 'seed'], axis=1)
        papa = pd.DataFrame(opa.iloc[:, -1][opa.index.get_level_values('design') == ds])
        papa.columns = ['ece']
        papa['classifier'] = cf
    else:
        results = pd.read_csv(OUTPUT_DIR + 'output_sharp_binning_' + dset + '.csv')
        results.drop(['dataset'], axis=1, inplace=True)
        opa = results[results.classifier == cf].copy()
        opa.drop(['classifier'], axis=1, inplace=True)
        opa = opa.groupby(['calibrator', 'design']).mean().drop(['Unnamed: 0', 'seed'], axis=1)
        papa = pd.DataFrame(opa.iloc[:, -1][opa.index.get_level_values('design') == ds])
        papa.columns = ['sharp']
        papa['classifier'] = cf
    return papa


dsets = ['bank', 'credit']
cfs = ['rf', 'xb', 'lr']
ds = 1

df_summary = pd.DataFrame()

for dset in dsets:
    for cf in cfs:
        scratch = parse_results(dset, cf, ds)
        scratch['dset'] = dset
        df_summary = pd.concat((df_summary, scratch))

df_summary.reset_index(inplace = True)
df_summary.drop(['design'], axis=1, inplace=True)
df_pivot = df_summary.pivot_table(columns=['dset', 'classifier'], index='calibrator', values='ece')
df_pivot['average'] = df_pivot.mean(axis=1)

f = open(OUTPUT_DIR + '/tex/bank_credit.tex', "w")
f.write(df_pivot.style.format("{:.3f}").to_latex(
        column_format='l|rrr|rrr|r|',
        caption='ECE for the bank and credit datasets, '
                  'with competing calibration algorithms '
                  'applied to the underlying classifiers'))
f.close()


dsets = ['churn', 'fetal']
cfs = ['rf', 'xb', 'lr']
ds = 1

df_summary = pd.DataFrame()

for dset in dsets:
    for cf in cfs:
        scratch = parse_results(dset, cf, ds)
        scratch['dset'] = dset
        df_summary = pd.concat((df_summary, scratch))


df_summary.reset_index(inplace = True)
df_summary.drop(['design'], axis=1, inplace=True)
df_pivot = df_summary.pivot_table(columns=['dset', 'classifier'], index='calibrator', values='ece')
df_pivot['average'] = df_pivot.mean(axis=1)

f = open(OUTPUT_DIR + '/tex/churn_fetal.tex', "w")
f.write(df_pivot.style.format("{:.3f}").to_latex(
        column_format='l|rrr|rrr|r|',
        caption = 'ECE for the churn and fetal datasets, '
                  'with competing calibration algorithms '
                  'applied to the underlying classifiers'))
f.close()

dsets = ['bank', 'credit']
cfs = ['rf', 'xb', 'lr']
ds = 1

df_summary = pd.DataFrame()

for dset in dsets:
    for cf in cfs:
        scratch = parse_results(dset, cf, ds)
        scratch['dset'] = dset
        df_summary = pd.concat((df_summary, scratch))

df_summary.reset_index(inplace = True)
df_summary.drop(['design'], axis=1, inplace=True)
df_pivot = df_summary.pivot_table(columns=['dset', 'classifier'], index='calibrator', values='ece')
df_pivot['average'] = df_pivot.mean(axis=1)

f = open(OUTPUT_DIR + '/tex/bank_credit.tex', "w")
f.write(df_pivot.style.format("{:.3f}").to_latex(
        column_format='l|rrr|rrr|r|',
        caption='ECE for the bank and credit datasets, '
                  'with competing calibration algorithms '
                  'applied to the underlying classifiers'))
f.close()


dsets = ['churn', 'fetal']
cfs = ['rf', 'xb', 'lr']
ds = 1

df_summary = pd.DataFrame()

for dset in dsets:
    for cf in cfs:
        scratch = parse_results(dset, cf, ds, 'sharp')
        scratch['dset'] = dset
        df_summary = pd.concat((df_summary, scratch))


df_summary.reset_index(inplace = True)
df_summary.drop(['design'], axis=1, inplace=True)
df_pivot = df_summary.pivot_table(columns=['dset', 'classifier'], index='calibrator', values='sharp')
df_pivot['average'] = df_pivot.mean(axis=1)

f = open(OUTPUT_DIR + '/tex/churn_fetal_sharp.tex', "w")
f.write(df_pivot.style.format("{:.3f}").to_latex(
        column_format='l|rrr|rrr|r|',
        caption = 'Sharpness for the churn and fetal datasets, '
                  'with competing calibration algorithms '
                  'applied to the underlying classifiers'))
f.close()

dsets = ['bank', 'credit']
cfs = ['rf', 'xb', 'lr']
ds = 1

df_summary = pd.DataFrame()

for dset in dsets:
    for cf in cfs:
        scratch = parse_results(dset, cf, ds, 'sharp')
        scratch['dset'] = dset
        df_summary = pd.concat((df_summary, scratch))

df_summary.reset_index(inplace = True)
df_summary.drop(['design'], axis=1, inplace=True)
df_pivot = df_summary.pivot_table(columns=['dset', 'classifier'], index='calibrator', values='sharp')
df_pivot['average'] = df_pivot.mean(axis=1)

f = open(OUTPUT_DIR + '/tex/bank_credit_sharp.tex', "w")
f.write(df_pivot.style.format("{:.3f}").to_latex(
        column_format='l|rrr|rrr|r|',
        caption='Sharpness for the bank and credit datasets, '
                  'with competing calibration algorithms '
                  'applied to the underlying classifiers'))
f.close()