# -*- coding: utf-8 -*-
"""grid_search_vggs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/FalsoMoralista/Notebooks/blob/master/notebooks/grid_search_vggs.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
# -*- coding: utf-8 -*-
"""GridSearchCV_PlantCLEF2013.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aZBAmZMLizQ6dA9POQHLiQ-a7AHY0Fxc
"""

#from sklearn import metrics
import pandas as pd
import numpy as np
#import glob
from sklearn.decomposition import PCA
from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import cross_validate
#from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import time
import collections
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#from collections import defaultdict

# Commented out IPython magic to ensure Python compatibility.
#from google.colab import drive
#drive.mount('/gdrive/')
# %cd /gdrive/My\ Drive/artigo_plantas
#!ls

def multi_clf_metrics(arq_csv):

	# Cria X (features) e y (labels)
  especies = pd.read_csv(arq_csv, low_memory=False)
  X = np.array(especies.iloc[2:,0:4096])
  y = np.array(especies.iloc[2:,4096])		
  # Executa o PCA
  pca = PCA(.90)
  principal_components = pca.fit_transform(X)
 # 'SVC - (linear, ovr)': OneVsRestClassifier(svm.SVC(kernel='linear', degree=3, coef0=0.0,
#											shrinking=True, probability=False, tol=0.001, cache_size=500,
#											class_weight=None, verbose=False, max_iter=-1,
#											decision_function_shape='ovr', random_state=None)),					'SVC - (linear, ovo)': svm.SVC(kernel='linear', degree=3, coef0=0.0,
#											shrinking=True, probability=False, tol=0.001, cache_size=500,
#											class_weight=None, verbose=False, max_iter=-1,
#											decision_function_shape='ovo', random_state=None),					'SVC - (polynomial, ovo)': svm.SVC(kernel='poly', degree=3, coef0=0.0,
#											shrinking=True, probability=False, tol=0.001, cache_size=500,
#											class_weight=None, verbose=False, max_iter=-1,
#											decision_function_shape='ovo', random_state=None),
#					'SVC - (sigmoid, ovr)': OneVsRestClassifier(svm.SVC(kernel='sigmoid', degree=3,coef0=0.0,
#											shrinking=True, probability=False, tol=0.001, cache_size=500,
#											class_weight=None, verbose=False, max_iter=-1,
#											decision_function_shape='ovr', random_state=None)),
#					'SVC - (sigmoid, ovo)': svm.SVC(kernel='sigmoid', degree=3, coef0=0.0,
#											shrinking=True, probability=False, tol=0.001, cache_size=500,
#											class_weight=None, verbose=False, max_iter=-1,
#											decision_function_shape='ovo', random_state=None)
  classifiers = {
					'SVC - (polynomial, ovr)': OneVsRestClassifier(svm.SVC(kernel='poly', degree=3, coef0=0.0,
											shrinking=True, probability=False, tol=0.001, cache_size=2000,
											class_weight=None, verbose=False, max_iter=-1,
											decision_function_shape='ovr', random_state=None))
}

  scoring = ['f1_micro']
  # Parametros para rodar com o grid search
  parameters = {"estimator__C" : [0.01, 0.1, 10], 'estimator__gamma':[0.0001, 0.001, 0.01]}
  results = []
  f = open('gd_srch_results_'+arq_csv,'w')
  f.write('classifier name,'+'best parameters,'+'all results\n')
  f.close()
  for name_clf, clf in classifiers.items():
    print('Executando classificador'+ name_clf)
    scores = GridSearchCV(clf, parameters,cv=5, scoring=scoring, refit='f1_micro', return_train_score=True, n_jobs=-1, verbose=51)
    result = scores.fit(principal_components,y)
    results.append(result)
    print('Melhor configuração: '+result.best_params_)
    f = open('gd_srch_results_'+arq_csv,'w')
    f.write(name_clf+',')
    f.write(result.best_params_+',')
    f.write(result+'\n')
    f.close()
  return result

results = multi_clf_metrics('plantclef2013_vgg16.csv')