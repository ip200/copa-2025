from river import datasets
import numpy as np
import random

import calibration as cal


import pandas as pd

from river import drift, compose, feature_extraction
from river import linear_model, forest, ensemble, tree, naive_bayes, neighbors
from river import metrics, preprocessing

from protected_classification import ProtectedClassification, y_pred_encode, p_pred_encode
from sklearn.metrics import log_loss, brier_score_loss

OUTPUT_DIR = './results/section_5_3/'

# dsets = {
#     # 'Phishing': datasets.Phishing(),
#     'ImageSegments': datasets.ImageSegments(),
# }
#
# models = {
#     'ARF': forest.ARFClassifier(),
#     'SRP': ensemble.SRPClassifier(),
#     'ADWIN': ensemble.ADWINBaggingClassifier(forest.ARFClassifier()),
#     'Bagging': ensemble.BaggingClassifier(forest.ARFClassifier()),
#     'Hoeffding': tree.HoeffdingAdaptiveTreeClassifier(),
#     'EFTree': tree.ExtremelyFastDecisionTreeClassifier(),
#     'DriftSRP': drift.DriftRetrainingClassifier(
#         model=ensemble.SRPClassifier(),
#         drift_detector=drift.binary.DDM()),
#     'DriftARF': drift.DriftRetrainingClassifier(
#         model=forest.ARFClassifier(),
#         drift_detector=drift.binary.DDM())
# }
#
#
# seeds = [100, 101, 102, 103, 104]
#
# df_container = pd.DataFrame()
#
# for dset_name, dset in dsets.items():
#     for model_name, mod in models.items():
#         for random_seed in seeds:
#
#             print(dset_name + ' ' + model_name + ' ' + str(random_seed))
#
#             random.seed(random_seed)
#
#             dataset = dset
#
#             mod.clear()
#
#             if dset_name == 'Phishing':
#                 model = compose.Pipeline(
#                     preprocessing.StandardScaler(),
#                     mod
#                 )
#             else:
#                 model = compose.Pipeline(
#                     preprocessing.StandardScaler(),
#                     preprocessing.OneHotEncoder(),
#                     mod
#                 )
#
#             pc = ProtectedClassification()
#
#             scratch = []
#             metric_acc = {}
#             metric_acc['base'] = metrics.Accuracy()
#             metric_acc['prot'] = metrics.Accuracy()
#
#             metric_roc = {}
#             metric_roc['base'] = metrics.ROCAUC()
#             metric_roc['prot'] = metrics.ROCAUC()
#
#             logloss = {}
#             logloss['base'] = []
#             logloss['prot'] = []
#
#             brierloss = {}
#             brierloss['base'] = []
#             brierloss['prot'] = []
#
#             acc = {}
#             acc['base'] = []
#             acc['prot'] = []
#
#             y_preds = []
#             y_primes = []
#             y_test = []
#             p_preds = []
#             p_primes = []
#
#             counter = 0
#             classes = []
#             for x, y in dataset:
#                 counter += 1
#                 print('\r', "{:.0%}".format(counter / dataset.n_samples), end='')
#                 classes.append(y)
#                 classes_unique = np.unique(np.array(classes))
#
#                 # classifier prediction
#                 y_pred = model.predict_one(x)
#                 p_pred = model.predict_proba_one(x)
#
#                 # protected classifcation prediction
#                 p_prime, y_prime = pc.predict_proba_one(p_pred)
#
#                 # if it is not the first prediction and we have more than a single class in the stream
#                 if len(classes_unique) > 1 and len(p_prime) > 0:
#                     metric_acc['base'].update(y, y_pred)
#                     metric_acc['prot'].update(y, y_prime)
#                     if dataset.n_classes:
#                         if dataset.n_classes < 3:
#                             metric_roc['base'].update(y, y_pred)
#                             metric_roc['prot'].update(y, y_prime)
#
#                     logloss['base'].append(
#                         log_loss(y_pred_encode(y, classes_unique)[0], p_pred_encode(p_pred, classes_unique)[0]))
#                     logloss['prot'].append(
#                         log_loss(y_pred_encode(y, classes_unique)[0], p_pred_encode(p_prime, classes_unique)[0]))
#                     brierloss['base'].append(
#                         brier_score_loss(y_pred_encode(y, classes_unique)[0],
#                                          p_pred_encode(p_pred, classes_unique)[0]))
#                     brierloss['prot'].append(
#                         brier_score_loss(y_pred_encode(y, classes_unique)[0],
#                                          p_pred_encode(p_prime, classes_unique)[0]))
#
#                     y_preds.append(y_pred_encode(y_pred, classes_unique)[0])
#                     y_primes.append(y_pred_encode(y_prime, classes_unique)[0])
#                     p_preds.append(p_pred_encode(p_pred, classes_unique)[0])
#                     p_primes.append(p_pred_encode(p_prime, classes_unique)[0])
#
#                 # learn from the example
#                 model.learn_one(x, y)
#                 pc.learn_one(p_pred, y)
#
#             print('\r', metric_acc['base'])
#
#             scratch.append(['log_loss', np.mean(np.array(logloss['base'])), np.mean(np.array(logloss['prot']))])
#             scratch.append(['brier_loss', np.mean(np.array(brierloss['base'])), np.mean(np.array(brierloss['prot']))])
#             scratch.append(['accuracy', metric_acc['base'].get(), metric_acc['prot'].get()])
#             if dataset.n_classes:
#                 if dataset.n_classes < 3:
#                     scratch.append(['ROC', metric_roc['base'].get(), metric_roc['prot'].get()])
#                     scratch.append([
#                         'cal_error',
#                         cal.get_calibration_error(np.array(p_preds), np.argmax(np.array(y_preds), axis=1), debias=True),
#                         cal.get_calibration_error(np.array(p_primes), np.argmax(np.array(y_primes), axis=1), debias=True)])
#
#             scratch = pd.DataFrame(scratch, columns=['metric', 'base', 'protected'])
#             scratch['seed'] = random_seed
#             scratch['dataset'] = dset_name
#             scratch['model'] = model_name
#             df_container = pd.concat((df_container, scratch), axis=0)
#             df_container.to_csv(OUTPUT_DIR + 'results.csv')


df_summary = pd.read_csv(OUTPUT_DIR + 'results.csv')

scratch = df_summary[df_summary.metric == 'log_loss']
scratch = scratch[scratch.dataset == 'Phishing']
scratch = scratch.drop(['metric'], axis=1)
tbl_1 = scratch.pivot_table(columns=['dataset'], index='model', values=['base', 'protected']).copy()

scratch = df_summary[df_summary.metric == 'log_loss']
scratch = scratch[scratch.dataset == 'ImageSegments']
scratch = scratch.drop(['metric'], axis=1)
tbl_2 = scratch.pivot_table(columns=['dataset'], index='model', values=['base', 'protected']).copy()

results_summary = pd.concat((tbl_1, tbl_2), axis=1)
results_summary.columns = pd.MultiIndex.from_product([['Phishing', 'ImageSegments'], ['base', 'protected']])
results_summary.loc['Average'] = results_summary.mean()

print(results_summary)

f = open(OUTPUT_DIR + '/tex/results.tex', "w")
f.write(results_summary.style.format("{:.4f}").to_latex(
        column_format='l|rr|rr|',
        caption='log loss'))
f.close()

