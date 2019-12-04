import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca_model(x, iris_dataset):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_data_frame = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    final_data_frame = pd.concat([principal_data_frame, iris_dataset[['target']]], axis=1)
    print("Creating the Plot for PCA ...\n")
    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title('2 Component Plot\nPCA successfully applied', fontsize=20)
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indices_to_keep = final_data_frame['target'] == target
        ax.scatter(final_data_frame.loc[indices_to_keep, 'PC1'],
                   final_data_frame.loc[indices_to_keep, 'PC2'],
                   c=color,
                   s=50)
    ax.legend(targets, loc='lower right', fancybox=True, shadow=True)
    ax.grid()
    plt.show()
    print("Plot has been successfully Created ...\n")
