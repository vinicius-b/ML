from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

iris = datasets.load_iris()
X = iris.data[:,:]
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

svm = SVC(kernel='linear', C=0.01)
svm.fit(X_train, Y_train)

rfc = RandomForestClassifier(max_depth=2, random_state=0)
rfc.fit(X_train, Y_train)


Y_pred = svm.predict(X_test)
Y_pred_rfc = rfc.predict(X_test)

svm_acc = round(accuracy_score(Y_test, Y_pred), 6)
svm_recall = round(recall_score(Y_test, Y_pred, average='weighted'), 6)
svm_precision = round(precision_score(Y_test, Y_pred, average='weighted'), 6)

rfc_acc = round(accuracy_score(Y_test, Y_pred_rfc), 6)
rfc_recall = round(recall_score(Y_test, Y_pred_rfc, average='weighted'), 6)
rfc_precision = round(precision_score(Y_test, Y_pred_rfc, average='weighted'), 6)

print '\n#### SVM model ####\n'

print 'Accuracy SVM: {}'.format(svm_acc)
print '---- Classification report SVM ----'
print classification_report(Y_test, Y_pred)
print '---- Confusion matrix SVM ----'
print confusion_matrix(Y_test, Y_pred)

print '\n#### RFC model ####\n'
print 'Accuracy RFC: {}'.format(rfc_acc)
print '---- Classification report RFC ----'
print classification_report(Y_test, Y_pred_rfc)
print '---- Confusion matrix RFC ----'
print confusion_matrix(Y_test, Y_pred_rfc)

cv_rfc = cross_val_score(rfc, X, Y, cv=10)
cv_svm = cross_val_score(svm, X, Y, cv=10)

print '\n---- Cross validation score ----'

print 'CV RFC score: {}'.format(cv_rfc.mean())
print 'CV SVM score: {}'.format(cv_svm.mean())

if cv_rfc.mean() > cv_svm.mean():
	print '\n ----------- RFC chosen based on cross validation score! -----------'
	parameters = {'max_depth':[1,10], 'random_state':[1, 10]}
	rfc_hps = GridSearchCV(rfc, parameters)
	rfc_hps.fit(X, Y)

	print '\n## Best parameters: {} ##'.format(rfc_hps.best_params_)

	Y_pred = rfc_hps.predict(X_test)

	rfc_acc = round(accuracy_score(Y_test, Y_pred), 6)
	rfc_recall = round(recall_score(Y_test, Y_pred, average='weighted'), 6)
	rfc_precision = round(precision_score(Y_test, Y_pred, average='weighted'), 6)

	print '\n#### Prediction after hyperparameter adjustment ####\n'
	print 'Accuracy: {}'.format(rfc_acc)
	print '---- Classification report ----'
	print classification_report(Y_test, Y_pred)
	print '---- Confusion Matrix ----'
	print confusion_matrix(Y_test, Y_pred)
else:
	print '\n ----------- SVM chosen based on cross validation score! -----------'
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	svm_hps = GridSearchCV(svm, parameters)
	svm_hps.fit(X, Y)

	print '\n## Best parameters: {} ##'.format(svm_hps.best_params_)

	Y_pred = svm_hps.predict(X_test)

	svm_acc = round(accuracy_score(Y_test, Y_pred), 6)
	svm_recall = round(recall_score(Y_test, Y_pred, average='weighted'), 6)
	svm_precision = round(precision_score(Y_test, Y_pred, average='weighted'), 6)

	print '\n#### Prediction after hyperparameter adjustment ####\n'
	print 'Accuracy: {}'.format(svm_acc)
	print '---- Classification report ----'
	print classification_report(Y_test, Y_pred)
	print '---- Confusion matrix ----'
	print confusion_matrix(Y_test, Y_pred)


