import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Отримання даних з датасету
# (лише 2 ознаки, довжина та ширина чашолистку)
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Ініціалізація об'єкту kmeans для кластеризації.
# n_clusters встановлює к-сть кластерів на основі
# максимального значення цільових міток з y
kmeans = KMeans(n_clusters=y.max() + 1, init='k-means++', n_init=10, max_iter=300,
                tol=0.0001, verbose=0, random_state=None, copy_x=True)

# Навчання моделі кластеризації
kmeans.fit(X)
# Виконання передбачення кластерів для вхідних даних
y_kmeans = kmeans.predict(X)

# Створюєтся графік з точками даних, де
# колір кожної точки відповідає прогнозованому
# кластеру (c=y_means)
# Також позначаються центри кластерів (centers)
# у вигляді чорних кругів
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# Функція створена для реалізації альтернативного методу пошуку кластерів
# Вона пприймає дані X та к-сть кластерів. Метод починає з вибору
# випадкових центроїдів, а далі виконує оновлення центрів кластерів
# на основі найближчих точок й перевіряє збіжність.
# Повертає центри та мітки кластерів
def find_clusters(X, n_clusters, rseed=2):
    # Випадковий вибір кластерів
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # Оголощення label, базуючись на найближчому центрі
        labels = pairwise_distances_argmin(X, centers)
        # Пошук нових центрів
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # Перевірка збіжності
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


# 3 різні візуалізації з різними параметрами початкового
# вибору центроїдів

centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()