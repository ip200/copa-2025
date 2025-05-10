from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


from protected_classification import ProtectedClassification
from venn_abers import VennAbersCalibrator

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from sklearn.calibration import CalibratedClassifierCV

import calibration as cal
# after https://pypi.org/project/uncertainty-calibration/

import warnings
warnings.filterwarnings('ignore')

import random

OUTPUT_DIR = './results/section_5_1_1/'

random_seed = 101
random.seed(random_seed)

# number of classes
no_classes = [2, 3, 5, 10]


train_samples = 1000
train_validation_samples = 1000 + train_samples
full_samples = 1000 + train_validation_samples


# Brier loss functions:
def Brier(y, p):
    loss = 0
    for i in range(K):
        if y == i:
            loss += (1 - p[i]) ** 2
        else:
            loss += p[i] ** 2
    return loss


def y_encode(y):
    y_encoded = np.zeros((y.size, y.max() + 1), dtype=int)
    y_encoded[np.arange(y.size), y] = 1
    return y_encoded


def Brier_loss(y, p):
    loss = np.mean(np.sum((y - p) ** 2, axis=1))
    return loss


# Arithmetic average of numbers given on the log10 scale:
def log_mean(x):
    m = np.max(x)
    return m + np.log10(np.mean(np.exp(np.log(10) * (x - m))))


def calc_losses(p_pred, y_test, p_prime):
    y_test_encoded = y_encode(y_test)
    accuracy_base = accuracy_score(y_test_encoded, p_pred > 1 / K)
    loss_brier_base = Brier_loss(y_test_encoded, p_pred)
    loss_log_base = log_loss(y_test, p_pred)
    f1_base = f1_score(y_test_encoded, p_pred > 1 / K, average='micro')
    roc_auc_base = roc_auc_score(y_test, p_pred[:,1] if K==2 else p_pred, multi_class='ovr')
    cal_error_base = cal.get_calibration_error(p_pred, y_test)

    accuracy_prot = accuracy_score(y_test_encoded, p_prime > 1 / K)
    loss_brier_prot = Brier_loss(y_test_encoded, p_prime)
    loss_log_prot = log_loss(y_test, p_prime)
    f1_prot = f1_score(y_test_encoded, p_prime > 1 / K, average='micro')
    roc_auc_prot = roc_auc_score(y_test, p_prime[:,1] if K==2 else p_prime, multi_class='ovr')
    cal_error_prot = cal.get_calibration_error(p_prime, y_test)


    df_list = [
        accuracy_base,
        accuracy_prot,
        loss_brier_base,
        loss_brier_prot,
        loss_log_base,
        loss_log_prot,
        roc_auc_base,
        roc_auc_prot,
        f1_base,
        f1_prot,
        cal_error_base,
        cal_error_prot
    ]

    df_stat = pd.DataFrame(df_list).T

    df_stat.columns = [
        'accuracy_base',
        'accuracy_prot',
        'base brier',
        'prot brier',
        'base log',
        'prot log',
        'base auc',
        'prot auc',
        'base f1',
        'prot f1',
        'base calibration',
        'prot calibration'
    ]

    return df_stat


lr = LogisticRegression()
gnb = GaussianNB()
svc = CalibratedClassifierCV(LinearSVC(C=1.0))
rfc = RandomForestClassifier()
xg = GradientBoostingClassifier()

clf_list = [
    (lr, "Logistic"),
    (gnb, "Naive Bayes"),
    (svc, "SVM"),
    (rfc, "Random forest"),
    (xg, "XGBoost"),
    (CalibratedClassifierCV(lr, cv=2, method="isotonic"), "Logistic - isotonic"),
    (CalibratedClassifierCV(gnb, cv=2, method="isotonic"), "Naive Bayes - isotonic"),
    (CalibratedClassifierCV(svc, cv=2, method="isotonic"), "SVM - isotonic"),
    (CalibratedClassifierCV(rfc, cv=2, method="isotonic"), "Random forest - isotonic"),
    (CalibratedClassifierCV(xg, cv=2, method="isotonic"), "XGBoost - isotonic"),
    (CalibratedClassifierCV(lr, cv=2, method="sigmoid"), "Logistic - sigmoid"),
    (CalibratedClassifierCV(gnb, cv=2, method="sigmoid"), "Naive Bayes - sigmoid"),
    (svc, "SVC - sigmoid"),
    (CalibratedClassifierCV(rfc, cv=2, method="sigmoid"), "Random forest - sigmoid"),
    (CalibratedClassifierCV(xg, cv=2, method="sigmoid"), "XGBoost - sigmoid"),
    (VennAbersCalibrator(lr, inductive=False, n_splits=2, random_state=random_seed), "Logistic - venn_abers"),
    (VennAbersCalibrator(gnb,  inductive=False, n_splits=2, random_state=random_seed), "Naive Bayes - venn_abers"),
    (VennAbersCalibrator(svc,  inductive=False, n_splits=2, random_state=random_seed), "SVM - venn_abers"),
    (VennAbersCalibrator(rfc,  inductive=False, n_splits=2, random_state=random_seed), "Random forest - venn_abers"),
    (VennAbersCalibrator(xg,  inductive=False, n_splits=2, random_state=random_seed), "XGBoost - venn_abers"),

]

clf_labels = ['L', 'NB', 'SVM', 'RF', 'XGB']

def parse_df(df_input, df_name):
    df_input['index'] = df_input.index
    df = df_input.copy()
    df['dataset'] = df_name
    df['exchangable'] = df['index'].str.contains('exch')
    df['isotonic'] = df['index'].str.contains('isotonic')
    df['sigmoid'] = df['index'].str.contains('sigmoid')
    df['venn_abers'] = df['index'].str.contains('venn_abers')
    df['n_classes'] = K
    df['n_train'] = 1000
    cols = df.columns.tolist()
    df = df[cols[-6:] + cols[:-7]]
    df.index = [a[0] for a in df.index.str.split(pat="-")]
    return df



def generate_calibration_stats(clf_list, X_train, y_train, X_test, y_test, rand_int):

    df = pd.DataFrame()

    for i, (clf, name) in enumerate(clf_list):
        print(name)

        clf.fit(X_train, y_train)
        p_pred = clf.predict_proba(X_test)

        pc  = ProtectedClassification()
        p_prime = pc.predict_proba(test_probs=p_pred, y=y_test)

        df = pd.concat((df, calc_losses(p_pred, y_test, p_prime)))


    for i, (clf, name) in enumerate(clf_list):
        print(name + ' - exch')

        clf.fit(X_train, y_train)
        p_pred = clf.predict_proba(X_test)

        pc = ProtectedClassification()
        p_prime_exch = pc.predict_proba(test_probs=p_pred[rand_int], y=y_test[rand_int])

        df = pd.concat((df, calc_losses(p_pred[rand_int], y_test[rand_int], p_prime_exch)))


    df.index = [i[1] for i in clf_list] + [i[1] + '- exch' for i in clf_list]

    return df


# unperturbed
print('unp')
df_unperturbed = pd.DataFrame()
for K in no_classes:
    print(K)
    X, y = make_classification(
        n_samples=full_samples, n_classes=K, n_features=20, n_informative=K, n_redundant=K, random_state=random_seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=1000
    )

    rand_int = np.random.permutation(len(y_test))

    df = generate_calibration_stats(clf_list, X_train, y_train, X_test, y_test, rand_int)
    df_u = parse_df(df, 'unperturbed')
    df_unperturbed = pd.concat((df_unperturbed, df_u), axis=0)

df_unperturbed.to_csv(OUTPUT_DIR + 'df_unperturbed.csv')

# covariate shift
print('covariate shift')
df_cov_shift = pd.DataFrame()

for K in no_classes:
    print(K)


    X, y = make_classification(
        n_samples=full_samples, n_classes=K, n_features=20, n_informative=K, n_redundant=K, random_state=random_seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=1000
    )

    X_scratch = X_test.copy()

    X_scratch[:, 0] = -X_test[:, 0]

    X_test = np.vstack((X_test[:500], X_scratch[500:]))

    # y_test = np.hstack((y_test[:500], y_test[500:][X_test[500:, 0] < 0]))
    # X_test = np.vstack((X_test[:500], X_test[500:][X_test[500:, 0] < 0]))

    rand_int = np.random.permutation(len(y_test))

    df = generate_calibration_stats(clf_list, X_train, y_train, X_test, y_test, rand_int)
    df_cs = parse_df(df, 'cov_shift')
    df_cov_shift = pd.concat((df_cov_shift, df_cs), axis=0)

df_cov_shift.to_csv(OUTPUT_DIR + 'df_cov_shift.csv')


# label imbalance
print('label imbalance')
df_label_imbalance = pd.DataFrame()

for K in no_classes:
    print(K)
    X, y = make_classification(
        n_samples=full_samples, n_classes=K, n_features=20, n_informative=K, n_redundant=K, random_state=random_seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=1000
    )

    X_test = np.vstack((X_test[:500], X_test[500:][y_test[500:] == 0]))
    y_test = np.hstack((y_test[:500], y_test[500:][y_test[500:] == 0]))

    rand_int = np.random.permutation(len(y_test))

    df = generate_calibration_stats(clf_list, X_train, y_train, X_test, y_test, rand_int)
    df_yi = parse_df(df, 'label_imbalance')
    df_label_imbalance = pd.concat((df_label_imbalance, df_yi), axis=0)

df_label_imbalance.to_csv(OUTPUT_DIR + 'df_label_imbalance.csv')

# label shift
print('label_shift')
df_label_shift = pd.DataFrame()

for K in no_classes:
    print(K)
    X, y = make_classification(
        n_samples=full_samples, n_classes=K, n_features=20, n_informative=K, n_redundant=K, random_state=random_seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=1000
    )

    y_scratch = y_test.copy()

    y_scratch[y_test==0] = 1
    y_scratch[y_test==1] = 0

    y_test = np.hstack((y_test[:500], y_scratch[500:]))

    rand_int = np.random.permutation(len(y_test))

    df = generate_calibration_stats(clf_list, X_train, y_train, X_test, y_test, rand_int)
    df_ls = parse_df(df, 'label_shift')
    df_label_shift = pd.concat((df_label_shift, df_ls), axis=0)

df_label_shift.to_csv(OUTPUT_DIR + 'df_label_shift.csv')

# ----------------- write output to tex---------------------

def summarise_stat(df_css, n_class, exch_flag, stat_list):
    tester = df_css[
        (df_css.n_classes == n_class) &
        (df_css.exchangable == exch_flag)
        ].groupby(['n_train', 'classifier', 'calib']).mean()

    df_summ = pd.DataFrame()
    for n_tr in [1000]:
        st = tester.reset_index().loc[:, ['n_train', 'classifier', 'calib', stat_list[0]]].copy()
        st = st[st.n_train == n_tr]
        st = st.pivot(index='classifier', columns='calib', values=stat_list[0])
        st.columns = [str(i) + ' - st' for i in st.columns]

        pr = tester.reset_index().loc[:, ['n_train', 'classifier', 'calib', stat_list[1]]].copy()
        pr = pr[pr.n_train == n_tr].copy()
        pr = pr.pivot(index='classifier', columns='calib', values=stat_list[1])
        pr.columns = [str(i) + ' - pr' for i in pr.columns]

        scratch = pd.concat((st, pr), axis=1)
        scratch['n_train'] = n_tr
        df_summ = pd.concat((df_summ, scratch))

    return df_summ.reset_index().groupby(['classifier', 'n_train']).mean()


def format_csv(df_css):
    df_css['calib'] = 'base'
    df_css.loc[df_css.isotonic == True, 'calib'] = 'isotonic'
    df_css.loc[df_css.sigmoid == True, 'calib'] = 'sigmoid'
    df_css.loc[df_css.venn_abers == True, 'calib'] = 'venn_abers'
    df_css.loc[df_css['Unnamed: 0'] == 'SVC ', 'Unnamed: 0'] = 'SVM'
    df_css.loc[df_css['Unnamed: 0'] == 'SVM ', 'Unnamed: 0'] = 'SVM'
    df_css.loc[df_css['Unnamed: 0'] == 'Logistic ', 'Unnamed: 0'] = 'Logistic'
    df_css.loc[df_css['Unnamed: 0'] == 'Naive Bayes ', 'Unnamed: 0'] = 'Naive Bayes'
    df_css.loc[df_css['Unnamed: 0'] == 'Random forest ', 'Unnamed: 0'] = 'Random forest'
    df_css.loc[df_css['Unnamed: 0'] == 'XGBoost ', 'Unnamed: 0'] = 'XGBoost'
    df_css.columns = ['classifier'] + list(df_css.columns[1:])
    df_css.drop(['index'], axis=1, inplace=True)
    return df_css


def tablerise(df_css, exch_flag, stat_list, caption_text, min_row=True):
    scratch = pd.DataFrame()
    for n_class in [2, 3, 5, 10]:
        df = summarise_stat(df_css, n_class, exch_flag, stat_list)
        df = df.reset_index().groupby(['classifier']).mean()
        df.drop(['n_train'], axis=1, inplace=True)
        df.columns = pd.MultiIndex.from_product([['standard', 'protected'], ['b', 'i', 's', 'v']])
        df['classes'] = int(n_class)
        scratch = pd.concat([scratch, df])

    scratch.loc['Average'] = scratch.mean()

    scratch.iloc[-1, -1] = ''
    scratch = scratch.reset_index().groupby(['classes', 'classifier']).mean().copy()

    #  bold max values

    df_s = scratch.style.format("{:.3f}")
    if min_row:
        for row in scratch.index:
            col = scratch.loc[row].idxmin()
            # redo formatting for a specific cell
            df_s = df_s.format(lambda x: "\\textbf{" + f'{x:.3f}' + "}", subset=(row, col))
    else:
        for row in scratch.index:
            col = scratch.loc[row].idxmax()
            # redo formatting for a specific cell
            df_s = df_s.format(lambda x: "\\textbf{" + f'{x:.3f}' + "}", subset=(row, col))

    opa = df_s.format_index(precision=0).to_latex(
        hrules=True,
        column_format='l|l|rrrr|rrrr',
        multicol_align='c',
        caption=caption_text)
    opa = opa.replace('\n\\multirow', '\n\\midrule\n\\multirow')
    opa = opa.replace('\n & Average', '\\midrule\\ & Average')

    return opa


file_list = ['df_unperturbed', 'df_cov_shift', 'df_label_imbalance', 'df_label_shift']


stats_list={}
stats_list['brier'] = [['base brier', 'prot brier'], 'Brier loss']
stats_list['log'] = [['base log', 'prot log'], 'log loss']
stats_list['ece'] = [['base calibration', 'prot calibration'], 'ECE loss']

exch_list = {}
exch_list['exch'] = [True, ' Exchangable']
exch_list['non_exch'] = [False, ' Non-exchangable']


for file_item in file_list:
    df = pd.read_csv(OUTPUT_DIR + file_item + '.csv')
    df_css = format_csv(df)
    for stat, stat_item in stats_list.items():
        for exch, exch_item in exch_list.items():
            scratch = tablerise(df_css, exch_item[0], stat_item[0], stat_item[1] + exch_item[1])
            f = open(OUTPUT_DIR + '/tex/' + file_item + '_' + stat + '_' + exch + '.tex', "w")
            f.write(scratch)
            f.close()

