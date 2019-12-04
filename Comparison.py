import matplotlib.pyplot as plt


def explaining_result(dataset_name, output_dict):
    accuracy = []
    model_list = []

    if dataset_name == 'Boston':
        figure1 = plt.figure(figsize=(30, 12))
        ax = figure1.subplots(2, 3)
        figure1.suptitle('Boston Comparison Visualization', fontsize=22)
        figure2 = plt.figure(figsize=(20, 10))
        ax2 = figure2.subplots(1, 1)
        figure1.delaxes(ax[1][1])
        figure1.delaxes(ax[1][2])
        for model in output_dict.keys():
            model_list.append(model)
            accuracy.append(output_dict[model]['Accuracy'])
        ax2.bar(model_list, accuracy)
        ax2.set_ylim([0, 100])
        ax2.xaxis.label.set_visible(False)
        ax2.set_title('Boston\nModel Accuracy', fontsize=17)

        for model in output_dict.keys():
            if model.startswith('KNN'):
                ax[0][0].plot(output_dict[model]['value_list'], output_dict[model]['accuracy_list'],
                              linestyle='solid', label='Accuracy')
                max_accuracy = max(output_dict[model]['accuracy_list'])
                value = output_dict[model]['value_list'][output_dict[model]['accuracy_list'].index(max_accuracy)]
                ax[0][0].scatter(value, max_accuracy, c='r', s=25, label='Maximum')
                ax[0][0].set_title('Best K for KNN Regression\nBest k = ' + str(value), fontsize=17)
                ax[0][0].set_xlabel('K')
                ax[0][0].set_ylabel('Accuracy')
                ax[0][0].legend(loc='upper right', fancybox=True, shadow=True)
                ax[0][0].grid()
            elif model.startswith('Polynomial'):
                ax[0][1].plot(output_dict[model]['value_list'], output_dict[model]['accuracy_list'],
                              linestyle='solid', label='Accuracy')
                max_accuracy = max(output_dict[model]['accuracy_list'])
                value = output_dict[model]['value_list'][output_dict[model]['accuracy_list'].index(max_accuracy)]
                ax[0][1].scatter(value, max_accuracy, c='r', s=25, label='Maximum')
                ax[0][1].set_title('Best Polynomial Degree for Polynomial Regression\nBest Degree = ' + str(value),
                                   fontsize=17)
                ax[0][1].set_xlabel('Poly Degree')
                ax[0][1].set_ylabel('Accuracy')
                ax[0][1].legend(loc='lower right', fancybox=True, shadow=True)
                ax[0][1].grid()
            elif model.startswith('Support Vector'):
                ax[0][2].plot(output_dict[model]['value_list'], output_dict[model]['accuracy_list'],
                              linestyle='solid', label='Accuracy')
                max_accuracy = max(output_dict[model]['accuracy_list'])
                value = output_dict[model]['value_list'][output_dict[model]['accuracy_list'].index(max_accuracy)]
                ax[0][2].scatter(value, max_accuracy, c='r', s=25, label='Maximum')
                ax[0][2].set_title('Best C for Support vector Regressor\nBest C = ' + str(value), fontsize=17)
                ax[0][2].set_xlabel('C')
                ax[0][2].set_ylabel('Accuracy')
                ax[0][2].legend(loc='lower right', fancybox=True, shadow=True)
                ax[0][2].grid()
            elif model.startswith('Multi-Layer'):
                ax[1][0].plot(range(len(output_dict[model]['loss'])), output_dict[model]['loss'],
                              linestyle='solid', label='Accuracy')
                ax[1][0].set_title('Loss Curve For MLP Regression', fontsize=17)
                ax[1][0].set_xlabel('Iteration')
                ax[1][0].set_ylabel('Loss')
                ax[1][0].legend(loc='upper right', fancybox=True, shadow=True)
                ax[1][0].grid()

        plt.show()

    elif dataset_name == 'BreastCancer':

        figure1 = plt.figure(figsize=(30, 15))
        ax = figure1.subplots(2, 3)
        figure1.suptitle('Breast Cancer Comparison Visualization', fontsize=22)
        figure2 = plt.figure(figsize=(20, 10))
        ax2 = figure2.subplots(1, 1)
        figure1.delaxes(ax[1][2])
        for model in output_dict.keys():
            model_list.append(model)
            accuracy.append(output_dict[model]['Accuracy'])
        ax2.bar(model_list, accuracy)
        ax2.set_ylim([0, 100])
        ax2.xaxis.label.set_visible(False)
        ax2.set_title('Breast Cancer\nModel Accuracy', fontsize=17)

        for model in output_dict.keys():
            if model.startswith('KNN'):
                ax[0][0].plot(output_dict[model]['value_list'], output_dict[model]['accuracy_list'],
                              linestyle='solid', label='Accuracy')
                max_accuracy = max(output_dict[model]['accuracy_list'])
                value = output_dict[model]['value_list'][output_dict[model]['accuracy_list'].index(max_accuracy)]
                ax[0][0].scatter(value, max_accuracy, c='r', s=25, label='Maximum')
                ax[0][0].set_title('Best K for KNN Classifier\nBest k = ' + str(value), fontsize=17)
                ax[0][0].set_xlabel('K')
                ax[0][0].set_ylabel('Accuracy')
                ax[0][0].legend(loc='upper right', fancybox=True, shadow=True)
                ax[0][0].grid()
            elif model.startswith('Logistic'):
                ax[0][1].plot(output_dict[model]['value_list'], output_dict[model]['accuracy_list'],
                              linestyle='solid', label='Accuracy')
                max_accuracy = max(output_dict[model]['accuracy_list'])
                value = output_dict[model]['value_list'][output_dict[model]['accuracy_list'].index(max_accuracy)]
                ax[0][1].scatter(value, max_accuracy, c='r', s=25, label='Maximum')
                ax[0][1].set_title('Best C for Logistic Regression\nBest C = ' + str(value), fontsize=17)
                ax[0][1].set_xlabel('Poly Degree')
                ax[0][1].set_ylabel('Accuracy')
                ax[0][1].legend(loc='lower right', fancybox=True, shadow=True)
                ax[0][1].grid()
            elif model.startswith('Decision'):
                ax[0][2].plot(output_dict[model]['value_list'], output_dict[model]['accuracy_list'],
                              linestyle='solid', label='Accuracy')
                max_accuracy = max(output_dict[model]['accuracy_list'])
                value = output_dict[model]['value_list'][output_dict[model]['accuracy_list'].index(max_accuracy)]
                ax[0][2].scatter(value, max_accuracy, c='r', s=25, label='Maximum')
                ax[0][2].set_title('Best Depth for Decision Tree\nBest Depth = ' + str(value), fontsize=17)
                ax[0][2].set_xlabel('Depth')
                ax[0][2].set_ylabel('Accuracy')
                ax[0][2].legend(loc='lower right', fancybox=True, shadow=True)
                ax[0][2].grid()
            elif model.startswith('Random'):
                ax[1][0].plot(output_dict[model]['value_list'], output_dict[model]['accuracy_list'],
                              linestyle='solid', label='Accuracy')
                max_accuracy = max(output_dict[model]['accuracy_list'])
                value = output_dict[model]['value_list'][output_dict[model]['accuracy_list'].index(max_accuracy)]
                ax[1][0].scatter(value, max_accuracy, c='r', s=25, label='Maximum')
                ax[1][0].set_title('Best Depth for Random Forest Classifier\nBest Depth = ' + str(value), fontsize=17)
                ax[1][0].set_xlabel('Depth')
                ax[1][0].set_ylabel('Accuracy')
                ax[1][0].legend(loc='lower right', fancybox=True, shadow=True)
                ax[1][0].grid()
            elif model.startswith('Multi-Layer'):
                ax[1][1].plot(range(len(output_dict[model]['loss'])), output_dict[model]['loss'],
                              linestyle='solid', label='Accuracy')
                ax[1][1].set_title('Loss Curve For MLP Classifier', fontsize=17)
                ax[1][1].set_xlabel('Iteration')
                ax[1][1].set_ylabel('Loss')
                ax[1][1].legend(loc='upper right', fancybox=True, shadow=True)
                ax[1][1].grid()

        plt.show()

    elif dataset_name == 'Diabetes':

        figure1 = plt.figure(figsize=(30, 12))
        ax = figure1.subplots(1, 3)
        figure1.suptitle('Diabetes Comparison Visualization', fontsize=22)
        figure2 = plt.figure(figsize=(20, 10))
        ax2 = figure2.subplots(1, 1)
        for model in output_dict.keys():
            model_list.append(model)
            accuracy.append(output_dict[model]['Accuracy'])
        ax2.bar(model_list, accuracy)
        ax2.set_ylim([0, 100])
        ax2.xaxis.label.set_visible(False)
        ax2.set_title('Diabetes\nModel Accuracy', fontsize=17)

        for model in output_dict.keys():
            if model.startswith('KNN'):
                ax[0].plot(output_dict[model]['value_list'], output_dict[model]['accuracy_list'],
                           linestyle='solid', label='Accuracy')
                max_accuracy = max(output_dict[model]['accuracy_list'])
                value = output_dict[model]['value_list'][output_dict[model]['accuracy_list'].index(max_accuracy)]
                ax[0].scatter(value, max_accuracy, c='r', s=25, label='Maximum')
                ax[0].set_title('Best K for KNN Classifier\nBest K = ' + str(value), fontsize=17)
                ax[0].set_xlabel('K')
                ax[0].set_ylabel('Accuracy')
                ax[0].legend(loc='lower right', fancybox=True, shadow=True)
                ax[0].grid()
            elif model.startswith('Decision'):
                ax[1].plot(output_dict[model]['value_list'], output_dict[model]['accuracy_list'],
                           linestyle='solid', label='Accuracy')
                max_accuracy = max(output_dict[model]['accuracy_list'])
                value = output_dict[model]['value_list'][output_dict[model]['accuracy_list'].index(max_accuracy)]
                ax[1].scatter(value, max_accuracy, c='r', s=25, label='Maximum')
                ax[1].set_title('Best Depth for Decision Tree\nBest Depth = ' + str(value), fontsize=17)
                ax[1].set_xlabel('Depth')
                ax[1].set_ylabel('Accuracy')
                ax[1].legend(loc='lower right', fancybox=True, shadow=True)
                ax[1].grid()
            elif model.startswith('Multi-Layer'):
                ax[2].plot(range(len(output_dict[model]['loss'])), output_dict[model]['loss'],
                           linestyle='solid', label='Accuracy')
                ax[2].set_title('Loss Curve For MLP Classifier', fontsize=17)
                ax[2].set_xlabel('Iteration')
                ax[2].set_ylabel('Loss')
                ax[2].legend(loc='upper right', fancybox=True, shadow=True)
                ax[2].grid()

        plt.show()

    elif dataset_name == 'MNIST':

        figure1 = plt.figure(figsize=(15, 8))
        ax = figure1.subplots(1, 1)
        figure1.suptitle('MNIST Comparison Visualization', fontsize=22)
        figure2 = plt.figure(figsize=(20, 10))
        ax2 = figure2.subplots(1, 1)
        for model in output_dict.keys():
            model_list.append(model)
            accuracy.append(output_dict[model]['Accuracy'])
        ax2.bar(model_list, accuracy)
        ax2.set_ylim([0, 100])
        ax2.xaxis.label.set_visible(False)
        ax2.set_title('Diabetes\nModel Accuracy', fontsize=17)

        for model in output_dict.keys():
            if model.startswith('Multi-Layer'):
                ax.plot(range(len(output_dict[model]['loss'])), output_dict[model]['loss'],
                        linestyle='solid', label='Accuracy')
                ax.set_title('Loss Curve For MLP Classifier', fontsize=17)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                ax.legend(loc='upper right', fancybox=True, shadow=True)
                ax.grid()

        plt.show()
