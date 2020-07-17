from sklearn import metrics
import pandas as pd
import numpy as np
import glob
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import time
import collections
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import defaultdict

#Variáveis do Dataset de Resultados
arquivo_csv = []
classifier = []
accuracy = []
std_accuracy = []

# Metrics Macro
precision_macro = []
std_precision_macro = []
recall_macro = []
std_recall_macro = []
f1_macro = []
std_f1_macro = []

# Metrics Micro
precision_micro = []
std_precision_micro = []
recall_micro = []
std_recall_micro = []
f1_micro = []
std_f1_micro = []

# Metrics Weighted
precision_weighted = []
std_precision_weighted = []
recall_weighted = []
std_recall_weighted = []
f1_weighted = []
std_f1_weighted = []

# Informações do PCA
n_samples = []
princ_compo = []
variancia = []

def report2dict(cr):
	# Parse rows
	tmp = list()
	for row in cr.split("\n"):
		parsed_row = [x for x in row.split("  ") if len(x) > 0]
		if len(parsed_row) > 0:
			tmp.append(parsed_row)

	# Store in dictionary
	measures = tmp[0]

	D_class_data = defaultdict(dict)
	for row in tmp[1:]:
		class_label = row[0]
		for j, m in enumerate(measures):
			D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
	return D_class_data

def multi_clf_metrics(arq_csv):

	# Cria X (features) e y (labels)
	especies = pd.read_csv(arq_csv, delimiter=',', low_memory=False)
	
	if arq_csv == 'plantclef2013.csv' or arq_csv == 'plantclef2013preclustering.csv':
		X = np.array(especies.iloc[2:,0:2048])
		y = np.array(especies.iloc[2:,2048])
    else if arq_csv == 'resnet50_features.csv' or arq_csv == 'resnet152v2_features.csv':
        X = np.array(especies.iloc[0:,0:2048])
        y = np.array(especies.iloc[0:,2048])
	else:
		X = np.array(especies.iloc[:,1:2049])
		y = np.array(especies['cluster'])
		
	# Executa o PCA
	pca = PCA(.90)
	principal_components = pca.fit_transform(X)

	classifiers = {'SVC - (linear, ovr)': OneVsRestClassifier(svm.SVC(kernel='linear', degree=3, coef0=0.0,
											shrinking=True, probability=False, tol=0.001, cache_size=200,
											class_weight=None, verbose=False, max_iter=-1,
											decision_function_shape='ovr', random_state=None)),
					'SVC - (linear, ovo)': svm.SVC(kernel='linear', degree=3, coef0=0.0,
											shrinking=True, probability=False, tol=0.001, cache_size=200,
											class_weight=None, verbose=False, max_iter=-1,
											decision_function_shape='ovo', random_state=None),
					'SVC - (polynomial, ovr)': OneVsRestClassifier(svm.SVC(kernel='poly', degree=3, coef0=0.0,
											shrinking=True, probability=False, tol=0.001, cache_size=200,
											class_weight=None, verbose=False, max_iter=-1,
											decision_function_shape='ovr', random_state=None)),
					'SVC - (polynomial, ovo)': svm.SVC(kernel='poly', degree=3, coef0=0.0,
											shrinking=True, probability=False, tol=0.001, cache_size=200,
											class_weight=None, verbose=False, max_iter=-1,
											decision_function_shape='ovo', random_state=None),
					'SVC - (sigmoid, ovr)': OneVsRestClassifier(svm.SVC(kernel='sigmoid', degree=3,coef0=0.0,
											shrinking=True, probability=False, tol=0.001, cache_size=200,
											class_weight=None, verbose=False, max_iter=-1,
											decision_function_shape='ovr', random_state=None)),
					'SVC - (sigmoid, ovo)': svm.SVC(kernel='sigmoid', degree=3, coef0=0.0,
											shrinking=True, probability=False, tol=0.001, cache_size=200,
											class_weight=None, verbose=False, max_iter=-1,
											decision_function_shape='ovo', random_state=None),
                    }

	scoring = ['accuracy',
				'precision_macro','recall_macro','f1_macro',
				'precision_micro','recall_micro','f1_micro',
				'precision_weighted','recall_weighted','f1_weighted']
    # Parametros para rodar com o grid search
    parameters = ['C':[0.001, 0.01, 0.1, 1, 5, 10, 25, 50, 100, 1000],
                  'gamma':[0.0001, 0.001, 0.01, 0.1, 1]]
	for name_clf, clf in classifiers.items():
		print(f'Executando classificador \033[1;31m{name_clf}\033[m')
		classifier.append(name_clf)
		arquivo_csv.append(arq_csv[:-4])
        # GridSearch para achar a melhor combinação de parâmetros (n_jobs=-1 para rodar em paralelo)
        scores = GridSearchCV(clf, parameters,cv=10,return_train_score=False, scoring=scoring, n_jobs=-1)
        scores.fit(principal_components,y)
    		
#		scores = cross_validate(clf, principal_components, y, scoring=scoring, cv=10, return_train_score=False)
#		y_pred = cross_val_predict(clf, principal_components, y, cv=10)
		df = pd.DataFrame(report2dict(classification_report(y, y_pred, target_names=sorted(set(y))))).T
		df.to_csv('classreport/'+arq_csv[:-4]+'_'+name_clf, index=True)
		df2 = pd.DataFrame(confusion_matrix(y, y_pred,), columns=sorted(set(y)), index=sorted(set(y)))
		df2.to_csv('matrix_confusion/'+arq_csv[:-4]+'_'+name_clf, index=True)
		
		accuracy.append(scores['test_accuracy'].mean())
		std_accuracy.append(scores['test_accuracy'].std())

		# Metrics Macro
		precision_macro.append(scores['test_precision_macro'].mean())
		std_precision_macro.append(scores['test_precision_macro'].std())
		recall_macro.append(scores['test_recall_macro'].mean())
		std_recall_macro.append(scores['test_recall_macro'].std())
		f1_macro.append(scores['test_f1_macro'].mean())
		std_f1_macro.append(scores['test_f1_macro'].std())
		print(f'\033[1;32mMétricas Macro extraidas!\033[m')

		# Metrics Micro
		precision_micro.append(scores['test_precision_micro'].mean())
		std_precision_micro.append(scores['test_precision_micro'].std())
		recall_micro.append(scores['test_recall_micro'].mean())
		std_recall_micro.append(scores['test_recall_micro'].std())
		f1_micro.append(scores['test_f1_micro'].mean())
		std_f1_micro.append(scores['test_f1_micro'].std())
		print(f'\033[1;32mMétricas Micro extraidas!\033[m')
		
		# Metrics Weighted
		precision_weighted.append(scores['test_precision_weighted'].mean())
		std_precision_weighted.append(scores['test_precision_weighted'].std())
		recall_weighted.append(scores['test_recall_weighted'].mean())
		std_recall_weighted.append(scores['test_recall_weighted'].std())
		f1_weighted.append(scores['test_f1_weighted'].mean())
		std_f1_weighted.append(scores['test_f1_weighted'].std())
		print(f'\033[1;32mMétricas Weighted extraidas!\033[m')
		
		n_samples.append(principal_components.shape[0])
		princ_compo.append(principal_components.shape[1])
		variancia.append(sum(pca.explained_variance_ratio_))

		print(f'Base {arq_csv[:-4]} \033[1;32mOK!\033[m')
		print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
		print('')

inicio = time.time()
for arq_csv in sorted(glob.glob('*.csv')):
	multi_clf_metrics(arq_csv)
fim = time.time()
horas = (fim-inicio) // 3600
print('Tempo de Execução foi de \033[1;31m{:.0f}\033[m horas'.format(horas))
'''print('')
print(f'arquivo_csv: {len(arquivo_csv)}')
print(f'classifier: {len(classifier)}')
print(f'accuracy: {len(accuracy)}')
print(f'std_accuracy: {len(std_accuracy)}')
print(f'precision_macro: {len(precision_macro)}')
print(f'std_precision_macro: {len(std_precision_macro)}')
print(f'recall_macro: {len(recall_macro)}')
print(f'std_recall_macro: {len(std_recall_macro)}')
print(f'f1_macro: {len(f1_macro)}')
print(f'std_f1_macro: {len(std_f1_macro)}')
print(f'precision_micro: {len(precision_micro)}')
print(f'std_precision_micro: {len(std_precision_micro)}')
print(f'recall_micro: {len(recall_micro)}')
print(f'std_recall_micro: {len(std_recall_micro)}')
print(f'f1_micro: {len(f1_micro)}')
print(f'std_f1_micro: {len(std_f1_micro)}')
print(f'precision_weighted: {len(precision_weighted)}')
print(f'std_precision_weighted: {len(std_precision_weighted)}')
print(f'recall_weighted: {len(recall_weighted)}')
print(f'std_recall_weighted: {len(std_recall_weighted)}')
print(f'f1_weighted: {len(f1_weighted)}')
print(f'std_f1_weighted: {len(std_f1_weighted)}')
print(f'n_samples: {len(n_samples)}')
print(f'princ_compo: {len(princ_compo)}')
print(f'variancia: {len(variancia)}')'''


# Criando um DataFrame com os Resultados
data = {'Dataset': arquivo_csv, 'Classificador': classifier, 'Accuracy': accuracy,'D.P. - Acurácia': std_accuracy, 
		'Precision Macro': precision_macro, 'D.P - Precision Macro': std_precision_macro,
		'Recall Macro': recall_macro, 'D.P - Recall Macro': std_recall_macro,
		'F1 Macro': f1_macro, 'D.P - F1 Macro': std_f1_macro,
		'Precision Micro': precision_micro, 'D.P - Precision Micro': std_precision_micro,
		'Recall Micro': recall_micro, 'D.P - Recall Micro': std_recall_micro,
		'F1 Micro': f1_micro, 'D.P - F1 Micro': std_f1_micro,
		'Precision Weighted': precision_weighted, 'D.P - Precision Weighted': std_precision_weighted,
		'Recall Weighted': recall_weighted, 'D.P - Recall Weighted': std_recall_weighted,
		'F1 Weighted': f1_weighted, 'D.P - F1 Weighted': std_f1_weighted,
		'N. Amostras': n_samples, 'PCA': princ_compo, 'Variância': variancia}
df = pd.DataFrame(data)
df.to_csv('resultados/resultados_mult_clf_metrics.csv', index=None)
