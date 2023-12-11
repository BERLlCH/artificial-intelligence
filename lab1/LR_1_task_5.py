import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()
print('Confusion matrix function from lib', confusion_matrix(df.actual_label.values, df.predicted_RF.values))


# Функції для підрахунку значень confusion matrix

# counts the number of true positives (y_true = 1, y_pred = 1)
def find_TP(y_true, y_pred):
 return sum((y_true == 1) & (y_pred == 1))
# counts the number of false negatives (y_true = 1, y_pred = 0)
def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))
# counts the number of false positives (y_true = 0, y_pred = 1)
def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))
# counts the number of true negatives (y_true = 0, y_pred = 0)
def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

print('\nTP:',find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:',find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:',find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:',find_TN(df.actual_label.values, df.predicted_RF.values))

# Загальна функція для пошуку значень confusion matrix:
def find_conf_matrix_values(y_true,y_pred):
 # calculate TP, FN, FP, TN
 TP = find_TP(y_true,y_pred)
 FN = find_FN(y_true,y_pred)
 FP = find_FP(y_true,y_pred)
 TN = find_TN(y_true,y_pred)
 return TP,FN,FP,TN

# Власна функція для дублювання confusion_matrix():
def berlinov_confusion_matrix(y_true, y_pred):
 TP, FN, FP, TN = find_conf_matrix_values(y_true,y_pred)
 return np.array([[TN,FP],[FN,TP]])

print('\nІстинно-негативний та хибно-позитивний (власна функція): ', berlinov_confusion_matrix(df.actual_label.values, df.predicted_RF.values)[0])
print('Невірно-негативний та істинно-позитивний (власна функція): ', berlinov_confusion_matrix(df.actual_label.values, df.predicted_RF.values)[1])

# Перевірка:
assert np.array_equal(berlinov_confusion_matrix(df.actual_label.values, df.predicted_RF.values), confusion_matrix(df.actual_label.values, df.predicted_RF.values)), 'berlinov_confusion_matrix() is not correct for RF'
assert np.array_equal(berlinov_confusion_matrix(df.actual_label.values, df.predicted_LR.values),confusion_matrix(df.actual_label.values, df.predicted_LR.values) ), 'berlinov_confusion_matrix() is not correct for LR'

print("\nAccuracy score function from lib: ", accuracy_score(df.actual_label.values, df.predicted_RF.values))

# Власна функція для дублювання accuracy_score():
def berlinov_accuracy_score(y_true, y_pred):
 # calculates the fraction of samples
 TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
 return (TP + TN) / (TP + TN + FP + FN)

# Перевірка:
assert berlinov_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values), 'berlinov_accuracy_score failed on RF'
assert berlinov_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), 'berlinov_accuracy_score failed on LR'

print('Accuracy RF (власна функція): %.3f'%(berlinov_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR (власна функція): %.3f'%(berlinov_accuracy_score(df.actual_label.values, df.predicted_LR.values)))


print('\nRecall score function from lib', recall_score(df.actual_label.values, df.predicted_RF.values))

# Власна функція, що дублює recall_score():
def berlinov_recall_score(y_true, y_pred):
 # calculates the fraction of positive samples predicted correctly
 TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
 return TP / (TP + FN)

# Перевірка:
assert berlinov_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values), 'my_accuracy_score failed on RF'
assert berlinov_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values, df.predicted_LR.values), 'my_accuracy_score failed on LR'
print('Recall RF (власна функція): %.3f'%(berlinov_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR (власна функція): %.3f'%(berlinov_recall_score(df.actual_label.values, df.predicted_LR.values)))


print('\nPrecision score function from lib', precision_score(df.actual_label.values, df.predicted_RF.values))

# Власна функція, що дублює precision_score():
def berlinov_precision_score(y_true, y_pred):
 # calculates the fraction of predicted positives samples that are actually positive
 TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
 return TP / (TP + FP)

# Перевірка:
assert berlinov_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values), 'my_accuracy_score failed on RF'
assert berlinov_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values, df.predicted_LR.values), 'my_accuracy_score failed on LR'
print('Precision RF (власна функція): %.3f'%(berlinov_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR (власна функція): %.3f'%(berlinov_precision_score(df.actual_label.values, df.predicted_LR.values)))

print('\nF1 score function from lib', f1_score(df.actual_label.values, df.predicted_RF.values))

# Власна функція, що дублює f1_score():
def berlinov_f1_score(y_true, y_pred):
 # calculates the F1 score
 recall = berlinov_recall_score(y_true,y_pred)
 precision = berlinov_precision_score(y_true,y_pred)
 return (2 * (precision * recall)) / (precision + recall)
# Перевірка:
assert berlinov_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values, df.predicted_RF.values), 'my_accuracy_score failed on RF'
assert berlinov_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values, df.predicted_LR.values), 'my_accuracy_score failed on LR'
print('F1 RF (власна функція): %.3f'%(berlinov_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR (власна функція): %.3f'%(berlinov_f1_score(df.actual_label.values, df.predicted_LR.values)))

# Зміна порогу:

print('\nscores with threshold = 0.5')
print('Accuracy RF: %.3f'%(berlinov_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f'%(berlinov_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f'%(berlinov_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f'%(berlinov_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f'%(berlinov_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f'%(berlinov_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f'%(berlinov_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f'%(berlinov_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)
plt.plot(fpr_RF, tpr_RF,'r-',label = '\nRF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values,
df.model_LR.values)
print('\nAUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

plt.plot(fpr_RF, tpr_RF,'r-',label = '\nRF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()