from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))

# Короткий опис набору даних
print(iris_dataset['DESCR'][:193] + "\n...")
print("Назви відповідей (сорти квітів): {}".format(iris_dataset['target_names']))
# Список рядків із описом кожної ознаки:
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))
#target та data. data – масив NumPy, який містить кількісні вимірювання
# довжини чашолистків, ширини чашолистків, довжини пелюсток та ширини пелюсток:
print("Тип масиву data: {}".format(type(iris_dataset['data'])))
# рядки - квітки ірису, стовпці - 4 ознаки, які були виміряні для кожної квітки:
print("Форма масиву data: {}".format(iris_dataset['data'].shape))

print('\nОзнаки для перших 5 прикладів:')
i = 0
while i < 5:
    print(iris_dataset.data[i])
    i += 1

print("Тип масиву target: {} ".format(type(iris_dataset['target'])))
print("Відповіді:\n{}".format(iris_dataset['target']))
