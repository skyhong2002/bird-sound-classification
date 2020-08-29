import numpy as np
import matplotlib.pyplot as plt
import itertools

import env

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 10)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [x[:5] + ".." for x in classes], rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == '__main__':
    class_names = env.classes.keys()
    cnf_matrix = np.load(f"{env.OUTPUT_DIR}/confusion_matrix.npy")
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                          title='confusion matrix')

    plt.savefig(f"{env.OUTPUT_DIR}/confusion_matrix.png", dpi=500)
    # plt.show()
    plt.close('all')
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='confusion matrix')

    plt.savefig(f"{env.OUTPUT_DIR}/confusion_matrix_norm.png", dpi=500)
    # plt.show()
