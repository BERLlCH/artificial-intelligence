import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Генерація даних
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X ** 2 + X + 2 + np.random.randn(m, 1)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Графік вхідних даних
plt.scatter(X, y, label='Дані', c='b', s=10)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Графік даних')
plt.legend(loc='best')
plt.show()

# Лінійна регресія
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X, y)

# Лінійна регресія
plt.scatter(X, y, color='green')
plt.plot(X, linear_regressor.predict(X), color='black', linewidth=1)
plt.title("Лінійна регресія")
plt.show()

# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=2, include_bias=False)
X_poly = polynomial.fit_transform(X)
polynomial.fit(X_poly, y)

print("\nX[0]: ", X[0])
print("X_poly: ", X_poly)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_poly, y)
y_pred = poly_linear_model.predict(X_poly)

print("\nR2 score: ", sm.r2_score(y, y_pred))
print("Interception: ", poly_linear_model.intercept_)
print("Coefficient: ", poly_linear_model.coef_)

# Поліноміальна регресія
plt.scatter(X, y, color='green')
plt.plot(X, y_pred, "*", color='black', linewidth=2)
plt.title("Поліноміальна регресія")
plt.show()