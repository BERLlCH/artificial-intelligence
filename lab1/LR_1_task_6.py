import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('data_metrics.csv')

X = data.drop('actual_label', axis=1)
y = data['actual_label']

# Розділяємо дані на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчання моделі SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Навчання моделі наївного байєсівського класифікатора
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Передбачення для SVM
svm_predictions = svm_model.predict(X_test)

# Передбачення для наївного байєсівського класифікатора
nb_predictions = nb_model.predict(X_test)

# Оцінка результатів SVM
print("Результати для SVM:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("\nДоповідь про классификацію:")
print(classification_report(y_test, svm_predictions))

# Оцінка результатів наївного байєсівського класифікатора
print("\nРезультати для наївного байєсівського класифікатора:")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("\nДоповідь про классификацію:")
print(classification_report(y_test, nb_predictions))