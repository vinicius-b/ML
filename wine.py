from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

wine = load_wine()
X = wine.data[:, :]
Y = wine.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

svm = SVC(kernel='linear', C=1)
svm.fit(X_train, Y_train)

Y_pred = svm.predict(X_test)

svm_acc = round(accuracy_score(Y_test, Y_pred), 6)
svm_recall = round(recall_score(Y_test, Y_pred, average='weighted'), 6)
svm_precision = round(precision_score(Y_test, Y_pred, average='weighted'), 6)

print '#### First Model ####\n'

print 'Accuracy: {}'.format(svm_acc)
print '---- Classification report ----'
print classification_report(Y_test, Y_pred)
print '---- Confusion Matrix ----'
print confusion_matrix(Y_test, Y_pred)

cv_results = cross_validate(svm, X, Y)

print '\n---- Cross Validation Score ----'
print cv_results['test_score']

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svm_hps = GridSearchCV(svm, parameters)
svm_hps.fit(X, Y)

print '\n## Best parameters: {} ##'.format(svm_hps.best_params_)

Y_pred = svm_hps.predict(X_test)

svm_acc = round(accuracy_score(Y_test, Y_pred), 6)
svm_recall = round(recall_score(Y_test, Y_pred, average='weighted'), 6)
svm_precision = round(precision_score(Y_test, Y_pred, average='weighted'), 6)

print '\n\n\n#### Prediction after hyperparameter adjustment ####\n'
print 'Accuracy: {}'.format(svm_acc)
print 'Recall: {}'.format(svm_recall)
print 'Precision: {}'.format(svm_precision)

print '---- Classification report ----'
print classification_report(Y_test, Y_pred)
print '---- Confusion Matrix ----'
print confusion_matrix(Y_test, Y_pred)

