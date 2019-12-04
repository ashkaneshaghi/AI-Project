# Internal Files Importing
import IrisPcaVisualization
import Datasets
import ModelEvaluation
import Comparison

# Importing Libraries
import math
import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, \
    f1_score, classification_report
import joblib as jl
import warnings
warnings.filterwarnings("ignore")


# Defining Global Variables
__model_path = 'Models/'
x_train_, x_test_, y_train_, y_test_ = '', '', '', ''


# Choosing the Dataset
def starting_point():
    print("<--- Available data sets list --->\n")
    print("[1] - Boston Dataset")
    print("[2] - BreastCancer Dataset")
    print("[3] - MNIST Dataset")
    print("[4] - Diabetes Dataset (Classification Version)")
    print("[5] = Iris Dataset")
    print("\n[0] = EXIT")
    dataset_num = input("\nEnter the number of Dataset you want to use : ")
    try:
        dataset = int(dataset_num)
        choose_dataset(dataset)
    except ValueError:
        print("\nInvalid input !!!\nPlease Enter a valid number (integer) ...\n")
        starting_point()


def choose_dataset(dataset_number):
    if dataset_number == 1:
        dataset = 'Boston'
        x_train, x_test, y_train, y_test = Datasets.load_boston()
        choose_model(dataset, x_train, x_test, y_train, y_test)
    elif dataset_number == 2:
        dataset = 'BreastCancer'
        x_train, x_test, y_train, y_test = Datasets.load_breast_cancer()
        choose_model(dataset, x_train, x_test, y_train, y_test)
    elif dataset_number == 3:
        dataset = 'MNIST'
        x_train, x_test, y_train, y_test = Datasets.load_mnist()
        choose_model(dataset, x_train, x_test, y_train, y_test)
    elif dataset_number == 4:
        dataset = 'Diabetes'
        x_train, x_test, y_train, y_test = Datasets.load_diabetes()
        choose_model(dataset, x_train, x_test, y_train, y_test)
    elif dataset_number == 5:
        x, iris_dataset = Datasets.load_iris()
        IrisPcaVisualization.pca_model(x, iris_dataset)
        starting_point()
    elif dataset_number == 0:
        exit(0)
    else:
        print("\nDataset you selected is not in the list !!!\nTry Again ...\n")
        starting_point()


# Choosing the model
def choose_model(dataset_name, x_train, x_test, y_train, y_test):
    global __model_path
    if dataset_name == 'Boston':
        output_dict = {}

        print("<---- Training With KNN Regression ---->")
        output_dict['KNN Regression'] = {}
        value_list = []
        accuracy_list = []
        print("\nCalculating the Best K has been successfully started ...")
        for k in range(1, 36):
            model = KNeighborsRegressor(n_neighbors=k)
            final_model, y_pred, training_score, testing_score = \
                ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
            value_list.append(k)
            accuracy_list.append(testing_score)
        output_dict['KNN Regression']['value_list'] = value_list
        output_dict['KNN Regression']['accuracy_list'] = accuracy_list
        print("Best K has been found ...")
        best_k = value_list[accuracy_list.index(max(accuracy_list))]
        model = KNeighborsRegressor(n_neighbors=best_k)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
        print("\nBest K = ", best_k, "\nModel has been successfully trained with k = ", best_k)
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        model_name = 'Boston_KNNReg_' + str(best_k) + '_' + str(time.time()) + '.sav'
        output_dict['KNN Regression']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['KNN Regression']['Model'] = model_name
        model_path = __model_path + "Boston/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n\n")

        print("<---- Training With Linear Regression ---->")
        output_dict['Linear Regression'] = {}
        model = LinearRegression()
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        model_name = 'Boston_LinReg_' + str(time.time()) + '.sav'
        output_dict['Linear Regression']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Linear Regression']['Model'] = model_name
        model_path = __model_path + "Boston/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n\n")

        print("<---- Training With Polynomial Regression ---->")
        output_dict['Polynomial Regression'] = {}
        value_list = []
        accuracy_list = []
        print("\nCalculating the Best Polynomial Degree has been successfully started ...")
        for degree in np.arange(1, 5):
            model = LinearRegression()
            final_model, y_pred, training_score, testing_score = \
                ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test, poly_deg=degree)
            if testing_score > 0:
                value_list.append(degree)
                accuracy_list.append(testing_score)
        print("Best Polynomial Degree has been found ...")
        best_degree = value_list[accuracy_list.index(max(accuracy_list))]
        output_dict['Polynomial Regression']['value_list'] = value_list
        output_dict['Polynomial Regression']['accuracy_list'] = accuracy_list
        model = LinearRegression()
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test, poly_deg=best_degree)
        print("\nBest Polynomial Degree = ", best_degree,
              "\nModel has been successfully trained with Polynomial Degree = ", best_degree)
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        model_name = 'Boston_LinRegPoly_' + str(best_degree) + '_' + str(time.time()) + '.sav'
        output_dict['Polynomial Regression']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Polynomial Regression']['Model'] = model_name
        model_path = __model_path + "Boston/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n\n")

        print("<---- Training With Support Vector Regression ---->")
        output_dict['Support Vector Regression'] = {}
        value_list = []
        accuracy_list = []
        print("\nCalculating the Best C has been successfully started ...")
        for c in range(1, 100):
            model = SVR(C=c)
            final_model, y_pred, training_score, testing_score = \
                ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test, scale=True)
            value_list.append(c)
            accuracy_list.append(testing_score)
        print("Best C has been found ...")
        best_c = value_list[accuracy_list.index(max(accuracy_list))]
        output_dict['Support Vector Regression']['value_list'] = value_list
        output_dict['Support Vector Regression']['accuracy_list'] = accuracy_list
        model = SVR(C=best_c)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test, scale=True)
        print("\nBest C = ", best_c,
              "\nModel has been successfully trained with C = ", best_c)
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        model_name = 'Boston_SVR_' + str(best_c) + '_' + str(time.time()) + '.sav'
        output_dict['Support Vector Regression']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Support Vector Regression']['Model'] = model_name
        model_path = __model_path + "Boston/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n\n")

        print("<---- Training With Multi-Layer Perceptron Regression ---->")
        output_dict['Multi-Layer Perceptron Regression'] = {}
        model = MLPRegressor(max_iter=2000, verbose=2)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test, scale=True)
        output_dict['Multi-Layer Perceptron Regression']['loss'] = final_model.loss_curve_
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        output_dict['Multi-Layer Perceptron Regression']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Multi-Layer Perceptron Regression']['Model'] = model_name
        model_name = 'Boston_MLPReg_' + str(time.time()) + '.sav'
        model_path = __model_path + "Boston/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n\n")

        Comparison.explaining_result(dataset_name, output_dict)
        starting_point()

    elif dataset_name == 'BreastCancer':
        output_dict = {}

        print("<---- Training With KNN Classifier ---->")
        output_dict['KNN Classifier'] = {}
        value_list = []
        accuracy_list = []
        print("\nCalculating the Best K has been successfully started ...")
        for k in range(1, 36):
            model = KNeighborsClassifier(n_neighbors=k)
            final_model, y_pred, training_score, testing_score = \
                ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
            value_list.append(k)
            accuracy_list.append(testing_score)
        print("Best K has been found ...")
        output_dict['KNN Classifier']['value_list'] = value_list
        output_dict['KNN Classifier']['accuracy_list'] = accuracy_list
        best_k = value_list[accuracy_list.index(max(accuracy_list))]
        model = KNeighborsClassifier(n_neighbors=best_k)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
        print("\nBest K = ", best_k, "\nModel has been successfully trained with k = ", best_k)
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        print("\nPrecision = ", precision_score(y_test, y_pred))
        print("Recall = ", recall_score(y_test, y_pred))
        print("F1_Score = ", f1_score(y_test, y_pred))
        model_name = 'Breast_Cancer_KNNCla_' + str(best_k) + '_' + str(time.time()) + '.sav'
        output_dict['KNN Classifier']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['KNN Classifier']['Model'] = model_name
        model_path = __model_path + "BreastCancer/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n")

        print("<---- Training With Logistic Regression ---->")
        output_dict['Logistic Regression'] = {}
        value_list = []
        accuracy_list = []
        print("\nCalculating the Best C has been successfully started ...")
        for c in range(1, 20):
            model = LogisticRegression(C=c, max_iter=10000, solver='lbfgs')
            final_model, y_pred, training_score, testing_score = \
                ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
            value_list.append(c)
            accuracy_list.append(testing_score)
        print("Best C has been found ...")
        output_dict['Logistic Regression']['value_list'] = value_list
        output_dict['Logistic Regression']['accuracy_list'] = accuracy_list
        best_c = value_list[accuracy_list.index(max(accuracy_list))]
        model = LogisticRegression(C=best_c, solver='lbfgs', max_iter=10000)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
        print("\nBest C = ", best_c, "\nModel has been successfully trained with C = ", best_c)
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        print("\nPrecision = ", precision_score(y_test, y_pred))
        print("Recall = ", recall_score(y_test, y_pred))
        print("F1_Score = ", f1_score(y_test, y_pred))
        print("Test ROC AUC score = ", roc_auc_score(y_test, y_pred))
        model_name = 'Breast_Cancer_LogReg_' + str(best_c) + '_' + str(time.time()) + '.sav'
        output_dict['Logistic Regression']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Logistic Regression']['Model'] = model_name
        model_path = __model_path + "BreastCancer/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n")

        print("<---- Training With Decision Tree ---->")
        output_dict['Decision Tree'] = {}
        value_list = []
        accuracy_list = []
        print("\nCalculating the Best Depth has been successfully started ...")
        for max_depth in range(1, 10):
            model = DecisionTreeClassifier(max_depth=max_depth)
            final_model, y_pred, training_score, testing_score = \
                ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
            value_list.append(max_depth)
            accuracy_list.append(testing_score)
        print("Best Depth has been found ...")
        output_dict['Decision Tree']['value_list'] = value_list
        output_dict['Decision Tree']['accuracy_list'] = accuracy_list
        best_max_depth = value_list[accuracy_list.index(max(accuracy_list))]
        model = DecisionTreeClassifier(max_depth=best_max_depth)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
        print("\nBest Depth = ", best_max_depth,
              "\nModel has been successfully trained with Depth = ", best_max_depth)
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        print("\nClassification Report = \n", classification_report(y_test, y_pred))
        model_name = 'Breast_Cancer_DecTreeCla_' + str(best_max_depth) + '_' + str(time.time()) + '.sav'
        output_dict['Decision Tree']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Decision Tree']['Model'] = model_name
        model_path = __model_path + "BreastCancer/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n")

        print("<---- Training With Random Forest Classifier ---->")
        output_dict['Random Forest Classifier'] = {}
        value_list = []
        accuracy_list = []
        print("\nCalculating the Best Depth has been successfully started ...")
        for max_depth in range(1, 10):
            model = RandomForestClassifier(max_depth=max_depth, n_estimators=100)
            final_model, y_pred, training_score, testing_score = \
                ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
            value_list.append(max_depth)
            accuracy_list.append(testing_score)
        print("Best Depth has been found ...")
        output_dict['Random Forest Classifier']['value_list'] = value_list
        output_dict['Random Forest Classifier']['accuracy_list'] = accuracy_list
        best_max_depth = value_list[accuracy_list.index(max(accuracy_list))]
        model = RandomForestClassifier(max_depth=best_max_depth, n_estimators=100)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
        print("\nBest Depth = ", best_max_depth,
              "\nModel has been successfully trained with Depth = ", best_max_depth)
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        model_name = 'Breast_Cancer_RanForCla_' + str(best_max_depth) + '_' + str(time.time()) + '.sav'
        output_dict['Random Forest Classifier']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Random Forest Classifier']['Model'] = model_name
        model_path = __model_path + "BreastCancer/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n")

        print("<---- Training With Multi-Layer Perceptron Classifier ---->")
        output_dict['Multi-Layer Perceptron Classifier'] = {}
        model = MLPClassifier(max_iter=10000, verbose=2)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test, scale=True)
        output_dict['Multi-Layer Perceptron Classifier']['loss'] = final_model.loss_curve_
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        print("\nClassification Report = \n", classification_report(y_test, y_pred))
        model_name = 'Breast_Cancer_MLPCla_' + str(final_model.n_iter_) + '_' + str(time.time()) + '.sav'
        output_dict['Multi-Layer Perceptron Classifier']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Multi-Layer Perceptron Classifier']['Model'] = model_name
        model_path = __model_path + "BreastCancer/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n")

        Comparison.explaining_result(dataset_name, output_dict)
        starting_point()

    elif dataset_name == 'MNIST':
        output_dict = {}

        print("Neural Network - Multi-layer Perceptron Classifier\n")
        epochs = 0
        learning_rate = 0.001
        hidden_layer_size = ()
        hls = 0
        try:
            hls = input("Inset Number of Layers you want : ")
            try:
                hls = int(hls)
                for i in range(hls):
                    size_of_layer = input("Enter Number of Neurons in Layer " + str(i + 1) + " --> : ")
                    try:
                        size_of_layer = int(size_of_layer)
                        hidden_layer_size = hidden_layer_size + (size_of_layer,)
                    except ValueError:
                        print("\nInvalid input !!!\nPlease Enter a valid number (integer) ...\n"
                              "Try Again from beginning\n")
                        choose_model(dataset_name, x_train, x_test, y_train, y_test)
            except ValueError:
                print("\nInvalid input !!!\nPlease Enter a valid number (integer) ...\nTry Again from beginning\n")
                choose_model(dataset_name, x_train, x_test, y_train, y_test)
        except ValueError:
            print("\nInvalid input !!!\nPlease Enter a valid number (integer) ...\nTry Again from beginning\n")
            choose_model(dataset_name, x_train, x_test, y_train, y_test)
        try:
            learning_rate = input("Enter Learning Rate Init you want (0.0001 - 0.2) : ")
            learning_rate = float(learning_rate)
        except ValueError:
            print("\nInvalid input !!!\nPlease Enter a valid number (integer) ...\nTry Again from beginning\n")
            choose_model(dataset_name, x_train, x_test, y_train, y_test)
        try:
            epochs = int(input("Enter Number of Iteration you want : "))
        except ValueError:
            print("\nInvalid input !!!\nPlease Enter a valid number (integer) ...\nTry Again from beginning\n")
            choose_model(dataset_name, x_train, x_test, y_train, y_test)
        alpha = 1e-4
        tol = 1e-4
        model = MLPClassifier(solver='sgd', hidden_layer_sizes=hidden_layer_size, alpha=alpha, tol=tol,
                              learning_rate_init=learning_rate, max_iter=epochs, random_state=1, warm_start=False,
                              verbose=10, batch_size=256)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
        output_dict['Multi-Layer Perceptron Classifier'] = {}
        output_dict['Multi-Layer Perceptron Classifier']['loss'] = final_model.loss_curve_
        print("Training has been successfully finished ...")
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        print("\nClassification Report = \n", classification_report(y_test, y_pred))
        model_name = 'Mnist_MLPCla_sgd_ReLU_' + str(hls) + '_' + str(final_model.n_iter_) + '_' + str(time.time()) + \
                     '.sav'
        output_dict['Multi-Layer Perceptron Classifier']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Multi-Layer Perceptron Classifier']['Model'] = model_name
        model_path = __model_path + "Mnist/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n")

        Comparison.explaining_result(dataset_name, output_dict)
        starting_point()

    elif dataset_name == 'Diabetes':
        output_dict = {}

        print("<---- Training With KNN Classifier ---->")
        output_dict['KNN Classifier'] = {}
        value_list = []
        accuracy_list = []
        print("\nCalculating the Best K has been successfully started ...")
        for k in range(1, 36):
            model = KNeighborsClassifier(n_neighbors=k)
            final_model, y_pred, training_score, testing_score = \
                ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
            value_list.append(k)
            accuracy_list.append(testing_score)
        print("Best K has been found ...")
        output_dict['KNN Classifier']['value_list'] = value_list
        output_dict['KNN Classifier']['accuracy_list'] = accuracy_list
        best_k = value_list[accuracy_list.index(max(accuracy_list))]
        model = KNeighborsClassifier(n_neighbors=best_k)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test, scale=True)
        print("\nBest K = ", best_k, "\nModel has been successfully trained with k = ", best_k)
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        print("\nPrecision = ", precision_score(y_test, y_pred))
        print("Recall = ", recall_score(y_test, y_pred))
        print("F1_Score = ", f1_score(y_test, y_pred))
        model_name = 'Diabetes_KNNCla_' + str(best_k) + '_' + str(time.time()) + '.sav'
        output_dict['KNN Classifier']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['KNN Classifier']['Model'] = model_name
        model_path = __model_path + "Diabetes/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n")

        print("<---- Training With Decision Tree ---->")
        output_dict['Decision Tree'] = {}
        value_list = []
        accuracy_list = []
        print("\nCalculating the Best Depth has been successfully started ...")
        for max_depth in range(1, 10):
            model = DecisionTreeClassifier(max_depth=max_depth)
            final_model, y_pred, training_score, testing_score = \
                ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test, scale=True)
            value_list.append(max_depth)
            accuracy_list.append(testing_score)
        print("Best Depth has been found ...")
        output_dict['Decision Tree']['value_list'] = value_list
        output_dict['Decision Tree']['accuracy_list'] = accuracy_list
        best_max_depth = value_list[accuracy_list.index(max(accuracy_list))]
        model = DecisionTreeClassifier(max_depth=best_max_depth)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test)
        print("\nBest Depth = ", best_max_depth,
              "\nModel has been successfully trained with Depth = ", best_max_depth)
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        print("\nClassification Report = \n", classification_report(y_test, y_pred))
        model_name = 'Diabetes_DecTreeCla_' + str(best_max_depth) + '_' + str(time.time()) + '.sav'
        output_dict['Decision Tree']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Decision Tree']['Model'] = model_name
        model_path = __model_path + "Diabetes/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n")

        print("<---- Training With Multi-Layer Perceptron Classifier ---->")
        output_dict['Multi-Layer Perceptron Classifier'] = {}
        model = MLPClassifier(max_iter=10000, verbose=2)
        final_model, y_pred, training_score, testing_score = \
            ModelEvaluation.model_evaluation(model, x_train, y_train, x_test, y_test, scale=True)
        output_dict['Multi-Layer Perceptron Classifier']['loss'] = final_model.loss_curve_
        print("\nTraining Score = ", training_score)
        print("Testing Set Score = ", testing_score)
        print("\nClassification Report = \n", classification_report(y_test, y_pred))
        model_name = 'Diabetes_MLPCla_' + str(final_model.n_iter_) + '_' + str(time.time()) + '.sav'
        output_dict['Multi-Layer Perceptron Classifier']['Accuracy'] = percentage_accuracy(testing_score, 4)
        output_dict['Multi-Layer Perceptron Classifier']['Model'] = model_name
        model_path = __model_path + "Diabetes/" + model_name
        jl.dump(final_model, model_path)
        print("\nModel Saved on Your Disk.\nModel Name = " + str(model_name) +
              "\nLocation = ProjectAI/" + str(model_path) + "\n")

        Comparison.explaining_result(dataset_name, output_dict)
        starting_point()


def percentage_accuracy(number, digits) -> float:
    stepper = 10.0 ** digits
    accuracy = (math.trunc(stepper * number) / stepper) * 100
    return accuracy


starting_point()
