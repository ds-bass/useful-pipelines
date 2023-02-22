# Homework 1
# to do's:
# add target name to values map and implement into viz and loops
# add viz or comps from training to validation metrics (what and how?)
#%%
import ast
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score,\
							precision_score, recall_score,\
							f1_score,\
							PrecisionRecallDisplay,\
							RocCurveDisplay, DetCurveDisplay,\
							confusion_matrix, ConfusionMatrixDisplay # other metrics?
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from itertools import product, cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# X and v as inputs
# X is data (must), v potential input of validation
# if v exists.... still do cross val on X, complete cross val, then finally predict on V
# currently works on binary classification, need to work on multiclass


most_important_metric = 'precision_score'
cross_val_train = True
validation_train = False
percent_data_for_val = 0.2
n_folds = 5
random_state = 77


assert ((validation_train == True) | (cross_val_train == True)), \
		"have to train on something!"
assert (most_important_metric in ['roc_auc_score', 'accuracy_score', 
								'precision_score', 'recall_score',
								'f1_score']), \
		"must pick error metric this was built for"
most_important_metric = most_important_metric.replace('_score', '')


def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average=average)


data = datasets.load_breast_cancer() #binary classification
# data = datasets.load_iris() #multiclass classification
M = np.array(data['data'])
L = np.array(data['target'])
features = data['feature_names']
target_names = data['target_names']


#look into these multi class scores more to make sure its what would be liked in general
if len(target_names) > 2:
    multi_class = True
    score_avg = 'weighted'
    roc_class = 'ovr'
else:
    multi_class = False
    score_avg = 'macro'
    roc_class = 'raise'


clfsList = [RandomForestClassifier,
			XGBClassifier#,
			# LogisticRegression
			]
clf_hyper = {
	'RandomForestClassifier': {
		"n_jobs": [1, 2, 5],
		"max_depth": [5, 10, 15],
		"min_samples_leaf": [1, 2, 4], 
		"random_state": [random_state]
	},
	'XGBClassifier': {
		"n_jobs": [1, 2, 5],
		"max_depth": [5, 10, 15],
		"learning_rate": [0.1, 0.01, 0.05],
		"gamma": [0, 0.25, 1],
		"random_state": [random_state]
	}#,
	# 'LogisticRegression': {
	# 	"tol": [.001, .01, .1],
	# 	"n_jobs": [1, 5, 10]}
 }


if cross_val_train == False:
	n_folds = 0

if validation_train == True:
	#split original data into train and validation
	idx = np.arange(M.shape[0])
	np.random.default_rng().shuffle(idx)
	idx1 = np.sort(idx[:int(M.shape[0] * percent_data_for_val)])
	idx2 = np.sort(idx[int(M.shape[0] * percent_data_for_val):])
	Mv = M[idx1, ...]
	M = M[idx2, ...]
	Lv = L[idx1, ...]
	L = L[idx2, ...]
else:
	Mv, Lv = None, None
	percent_data_for_val = 0.2

"""
	run ml algorithm w/given hyperparameters
	cross validation and/or regular validation hold out carried out
	returns dictionary of models metric results
"""
def run(a_clf, M, L, Mv, Lv, n_folds, cross_val_train, validation_train, clf_hyper={}):
	ret = {}
	if cross_val_train == True:
		kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
		
		for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
			clf = a_clf(**clf_hyper)
			clf.fit(M[train_index], L[train_index])
			pred = clf.predict(M[test_index])
			# import pdb; pdb.set_trace()
			if multi_class == True:
				# pred = np.argmax(pred, axis=1)
				ret[ids] = {
					'clf': clf,
					'accuracy': accuracy_score(L[test_index], pred),
					'f1': f1_score(L[test_index], pred, average=score_avg),
					'roc_auc': multiclass_roc_auc_score(L[test_index], pred),
					'precision': precision_score(L[test_index], pred, average=score_avg),
					'recall': recall_score(L[test_index], pred, average=score_avg)
				}
			else:
				ret[ids] = {
					'clf': clf,
					'accuracy': accuracy_score(L[test_index], pred),
					'f1': f1_score(L[test_index], pred, average=score_avg),
					'roc_auc': roc_auc_score(L[test_index], pred, average=score_avg, multi_class=roc_class),
					'precision': precision_score(L[test_index], pred, average=score_avg),
					'recall': recall_score(L[test_index], pred, average=score_avg)
				}
	
	if validation_train == True:
		clf = a_clf(**clf_hyper)
		clf.fit(M, L)
		pred = clf.predict(Mv)
		if multi_class == True:
			# pred = np.argmax(pred, axis=1)
			ret[n_folds] = {
				'clf': clf,
				'accuracy': accuracy_score(Lv, pred),
				'f1': f1_score(Lv, pred, average=score_avg),
				'roc_auc': multiclass_roc_auc_score(Lv, pred),
				'precision': precision_score(Lv, pred, average=score_avg),
				'recall': recall_score(Lv, pred, average=score_avg)
			}
		else:
			ret[n_folds] = {
				'clf': clf,
				'accuracy': accuracy_score([L[test_index], pred]),
				'f1': f1_score(L[test_index], pred, average=score_avg),
				'roc_auc': roc_auc_score(L[test_index], pred, average=score_avg, multi_class=roc_class),
				'precision': precision_score(L[test_index], pred, average=score_avg),
				'recall': recall_score(L[test_index], pred, average=score_avg)
			}
	
	return ret

model_results = {}


#Loop through the clfs List to run algorithm
for clfs in clfsList:
	model_name = str(clfs).rsplit('.',1)[1][:-2]
	print('running', model_name, 'models\ndatetime:', datetime.now(timezone('EST')).strftime('%Y-%m-%d %H:%M:%S'))
	
	list_of_list_hypers = [x for x in clf_hyper[model_name].values()]
	
	#loop through cartesian product of clf's hyperparameter arguements
	for arg_values in product(*list_of_list_hypers):
		clf_args = {}
		
		#build hyperparameter dictionary to unpack later when actually running the model
		for key, arg in zip(clf_hyper[model_name].keys(), arg_values):
			clf_args[key] = arg

		#fit and get model prediction results
		try: #some combos of arguments dont work together
			results = run(clfs, M, L, Mv, Lv, n_folds, cross_val_train, validation_train, clf_hyper=clf_args)

			#store model results for visualization later
			for k, v in results.items():
				#adding temp dictionary data
				temp_dict = {
					model_name: {
						str(clf_args): {
							'accuracy': [round(results[k]['accuracy'], 8)],
							'f1': [round(results[k]['accuracy'], 8)],
							'roc_auc': [round(results[k]['roc_auc'], 8)],
							'precision': [round(results[k]['precision'], 8)],
							'recall': [round(results[k]['recall'], 8)]
						}
					}
				}

				#first run through, append results dict
				if len(model_results) == 0:
					model_results = temp_dict.copy()
				else:
					#loop through models already ran to find where to put or append results
					for k1, v1 in model_results.items():
						#find previously ran model algo (ex XGB)
						if k1 == model_name:
							for k2, v2 in v1.items():
								#find previously ran hypers of given model algo and append other cross val results
								if k2 == str(clf_args):
									model_results[k1][k2]['accuracy'].append(temp_dict[model_name][str(clf_args)]['accuracy'][0])
									model_results[k1][k2]['f1'].append(temp_dict[model_name][str(clf_args)]['f1'][0])
									model_results[k1][k2]['roc_auc'].append(temp_dict[model_name][str(clf_args)]['roc_auc'][0])
									model_results[k1][k2]['precision'].append(temp_dict[model_name][str(clf_args)]['precision'][0])
									model_results[k1][k2]['recall'].append(temp_dict[model_name][str(clf_args)]['recall'][0])
							#if first of new hyperparameter batch being ran, make new value of model key
							if str(clf_args) not in v1.keys():
								model_results[k1][str(clf_args)] = temp_dict[model_name][str(clf_args)]
					#if first of model being ran, make new key for it
					if model_name not in model_results.keys():
						model_results[model_name] = temp_dict[model_name]
		except:
			pass

#%%
#Processing and aggregating of model results. 
	#1. pull out each stat we want to graph
		#Average the CV results for single metric
	#2. zip up each into a sortable array 
		#Might need to make an index with enumerate maybe. 
	#3. Sort them by chosen metric.  (accuracy, roc_auc, precision, recall)
	#4. Plot them.

models, hypers, accuracy, f1, roc_auc, precision, recall = [], [], [], [], [], [], []

for model in model_results:
	for k, v in model_results[model].items():
		models.append(model)
		hypers.append(k)
		accuracy.append(v['accuracy'])
		f1.append(v['f1'])
		roc_auc.append(v['roc_auc'])
		precision.append(v['precision'])
		recall.append(v['recall'])
	
res_dict = {
	'model': 0, 
	'hypers': 1,
	'accuracy': 2,
	'f1': 3,
	'roc_auc': 4,
	'precision': 5,
	'recall': 6
}

#Sort the data by chosen metric
all_data = sorted(zip(models, hypers, accuracy, f1, roc_auc, precision, recall),
				key=lambda x: np.mean(x[res_dict[most_important_metric]]),
				reverse=True)


#Get the top 5 accuracies for each model type
#filter the sorted data to only include the top 5 for each model type
top_models = []
for model in list(model_results.keys()):
	if model not in top_models:
		all_data_copy = all_data.copy()
		model_extract = filter(lambda x: x[res_dict['model']] == model, all_data_copy)
		top_sorted = sorted(model_extract, key=lambda x: np.mean(x[res_dict[most_important_metric]]), reverse=True)[:5]
		top_models.append(top_sorted)

#Plot distribution of error metrics for top 5 models per algorithm
#top 5 for each model plotting in a 2x2 grid.  
# Topleft = accuracy
# Topright = roc_auc
# Bottomleft = precision
# Bottomright = recall
if cross_val_train == True:
	for model in top_models:
		fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize=(10,9))
		plt.subplots_adjust(hspace=0.25)
		fig.suptitle(model[0][0], fontsize=16)
		ax[0, 0].set_title('Accuracy')
		ax[0, 1].set_title('F1')
		ax[1, 0].set_title('ROC AUC')
		ax[1, 1].set_title('Precision')
		ax[2, 0].set_title('Recall')

		ax[0, 0].set_yticklabels(range(1, len(model)+1))
		ax[0, 1].set_yticklabels(range(1, len(model)+1))
		ax[1, 0].set_yticklabels(range(1, len(model)+1))
		ax[1, 1].set_yticklabels(range(1, len(model)+1))
		ax[2, 0].set_yticklabels(range(1, len(model)+1))

		print(f'Top five parameters for {model[0][0]} are:')

		for i in range(len(model)):

			print(f'{i+1}: {model[i][1]}')
			
			ax[0, 0].boxplot(model[i][res_dict['accuracy']], 
							notch=False,
							sym = 'rs',
							vert=False,
							positions=[i+1]
					)
			ax[0, 1].boxplot(model[i][res_dict['f1']], 
							notch=False,
							sym='rs', 
							vert=False,
							positions=[i+1]
					)
			ax[1, 0].boxplot(model[i][res_dict['roc_auc']],
							notch=False, 
							sym='rs', 
							vert=False,
							positions=[i+1]
					)
			ax[1, 1].boxplot(model[i][res_dict['precision']], 
							notch=False, 
							sym='rs', 
							vert=False,
							positions=[i+1])
			ax[2, 0].boxplot(model[i][res_dict['recall']], 
							notch=False, 
							sym='rs', 
							vert=False,
							positions=[i+1])

		plt.show()
		print('Top params for ' + model[0][0] + ': ' + str(model[0][1]))
		print(f'With a {most_important_metric} training mean of:',
			round(np.mean(model[0][res_dict[most_important_metric]]), 4))
		print('\n\n')

# plt.subplots_adjust(hspace=0, vspace=0)

#%%
# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

# get best model from each model type and plot unique plots
# have to re-fit best model of each, so can try to fix that later
if validation_train == False:
	#if didnt split data before, split now for viz
	idx = np.arange(M.shape[0])
	np.random.default_rng().shuffle(idx)
	idx1 = np.sort(idx[:int(M.shape[0] * percent_data_for_val)])
	idx2 = np.sort(idx[int(M.shape[0] * percent_data_for_val):])
	Mv = M[idx1, ...]
	M = M[idx2, ...]
	Lv = L[idx1, ...]
	L = L[idx2, ...]

# %%
if multi_class == False:
	fig, [ax_roc, ax_prec, ax_det] = plt.subplots(3, 1, figsize=(6, len(top_models)*8))
else:
	fig, ax_prec = plt.subplots(1, 1, figsize=(6, len(top_models)*8))

#confusion matrices
f, axes = plt.subplots(1, len(top_models), figsize=(len(top_models)*8, 6), sharey='row')

# plot curves of each top model
for model in range(len(top_models)):
	for clf in clfsList:
		model_name = str(clf).rsplit('.',1)[1][:-2]
		
		if model_name == top_models[model][0][0]:
			clf = clf(**ast.literal_eval(top_models[model][0][1])) # unpack parameters
			clf.fit(M, L)
			pred = clf.predict(Mv)

			cm = confusion_matrix(Lv, pred)
			cm_display = ConfusionMatrixDisplay(cm, display_labels=target_names)
			cm_display.plot(ax=axes[model], xticks_rotation=45)
			cm_display.ax_.set_title(model_name)
			cm_display.im_.colorbar.remove()
			cm_display.ax_.set_xlabel('')
			if model!=0:
				cm_display.ax_.set_ylabel('')
			
			if multi_class == True:

				precision_multi, recall_multi = {}, {}
				for i in range(len(target_names)):

					idxs = np.where(Lv==i)[0]
					precision_multi[i], recall_multi[i], _ = precision_recall_curve(
						Lv[idxs], pred[idxs], pos_label=target_names[i]
					)
					# plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
				
				
				# precision, recall, avg_precision = {}, {}, {}
				# for i in range(len(target_names)):
				# 	idxs = np.where(Lv==i)[0]
				# 	precision[i], recall[i], _ = precision_recall_curve(
				# 		Lv[idxs], pred[idxs], pos_label=i)
				# 	avg_precision[i] = average_precision_score(
				# 		Lv[idxs], pred[idxs], pos_label=i)
				
				# precision['micro'], recall['micro'], _ = precision_recall_curve(
				# 	Lv, pred)
				# avg_precision['micro'] = average_precision_score(
				# 	Lv, pred, average='micro')
				
				# colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
				# fscores = np.linspace(0.2, 0.8, num=4)
				# lines, labels = [], []
				
				# for fscore in fscores:
				# 	x = np.linspace(0.01, 1)
				# 	y = fscore * x / (2 * x - fscore)
				# 	(l,) = plt.plot(x[y>=0], y[y>=0], color='gray', alpha=0.2)
				# 	plt.annotate("f1={0:0.1f}".format(fscore), xy=(0.9, y[45] + 0.2))
				
				# pr_display = PrecisionRecallDisplay(
				# 	recall=recall['micro'],
				# 	precision=precision['micro'],
				# 	average_precision=avg_precision['micro'])
				# pr_display.plot(ax=ax_prec, name='Micro-Avg', color='gold')

				
				# for i, color in zip(range(len(target_names)), colors):
				# 	PrecisionRecallDisplay(
				# 		recall=recall[i], 
				# 		precision=precision[i], 
				# 		average_precison=avg_precision[i])
				# 	pr_display.plot(ax=ax_prec, name=f'{target_names[i]}', color=color)

			else:
				RocCurveDisplay.from_estimator(clf, Mv, Lv, ax=ax_roc, name=model_name)
				PrecisionRecallDisplay.from_estimator(clf, Mv, Lv, ax=ax_prec, name=model_name)
				DetCurveDisplay.from_estimator(clf, Mv, Lv, ax=ax_det, name=model_name)

if multi_class == False:
	ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
	ax_prec.set_title("Precision Recall Tradeoff curves")
	ax_det.set_title("Detection Error Tradeoff (DET) curves")

	ax_roc.grid(linestyle="--")
	ax_prec.grid(linestyle="--")
	ax_det.grid(linestyle="--")
else:
	# handles, labels = pr_display.ax_.get_legend_handles_labels()
	# handles.extend([l])
	# labels.extend(['iso-f1 curves'])]
	for i in range(len(target_names)):
		ax_prec.plot(recall_multi[i], precision_multi[i], lw=2, label='class {}'.format(i))
	ax_prec.set_xlim(0.0, 1.05)
	ax_prec.set_ylim([0.0, 1.05])
	# ax_prec.legend(handles=handles, labels=labels, loc='best')

	ax_prec.set_title("Precision Recall Tradeoff curves")
	ax_prec.grid(linestyle="--")

f.text(0.4, 0.1, 'Predicted label', ha='left')
plt.subplots_adjust(wspace=0.40, hspace=0.1)
f.colorbar(cm_display.im_, ax=axes)

# plt.legend()
plt.show()

# %%
