import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
Loading all available Data sets with different function which is called from Main.py
'''


# Loading Data sets
def load_boston():
    print("\nLoading Boston Dataset ...")
    boston_dataset = datasets.load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    print("\nDone!!!\nSplitting Data ...")
    x = boston.iloc[:, 0:13]
    y = boston.iloc[:, 13]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    print("Splitting Data has been successfully finished ...\n")
    return x_train, x_test, y_train, y_test


def load_breast_cancer():
    print("\nLoading BreastCancer Dataset ...")
    breast_cancer_dataset = datasets.load_breast_cancer()
    breast_cancer = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
    breast_cancer['traget'] = breast_cancer_dataset.target
    print("\nDone!!!\nSplitting Data ...")
    x = breast_cancer.iloc[:, 0:30]
    y = breast_cancer.iloc[:, 30]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    print("Splitting Data has been successfully finished ...")
    return x_train, x_test, y_train, y_test


def load_mnist():
    print("\nLoading MNIT Dataset ...")
    data_path = 'Datasets'
    mnist = datasets.fetch_openml('mnist_784', data_home=data_path)
    print("\nDone!!!\nSplitting Data ...")
    x_train, x_test, y_train, y_test = train_test_split(mnist.data / 255.0, mnist.target.astype("int0"), test_size=0.33)
    print("Splitting Data has been successfully finished ...")
    print("\nNumber of Images in Training Set = ", x_train.shape[0])
    print("Number of Images in Testing Set = ", y_train.shape[0])
    pix = int(np.sqrt(x_train.shape[1]))
    print("Each Image is : ", pix, " by ", pix, "Pixels")
    return x_train, x_test, y_train, y_test


def load_diabetes():
    print("\nLoading Diabetes Dataset ...")
    diabetes_dataset = pd.read_csv('Datasets/diabetes.csv')
    print("\nDone!!!\n\nManipulating Data ...")
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for column in zero_not_accepted:
        diabetes_dataset[column] = diabetes_dataset[column].replace(0, np.NaN)
        mean = int(diabetes_dataset[column].mean(skipna=True))
        diabetes_dataset[column] = diabetes_dataset[column].replace(np.NaN, mean)
    print("Zeros Successfully Replaced ...\n\nSplitting Data...")
    x = diabetes_dataset.iloc[:, 0:8]
    y = diabetes_dataset.iloc[:, 8]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
    print("Splitting Data has been successfully finished ...")
    return x_train, x_test, y_train, y_test


def load_iris():
    print("\nLoading Iris Dataset ...")
    data_path = 'Datasets/iris.data'
    iris_dataset = pd.read_csv(data_path,
                               names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
    print("\nDone!!!\nManipulating Data ...")
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    x = iris_dataset.loc[:, features].values
    y = iris_dataset.loc[:, ['target']].values
    sc_x = StandardScaler()
    x = sc_x.fit_transform(x)
    print("Data has been Separated and Standardized Successfully ...\n")
    return x, iris_dataset

