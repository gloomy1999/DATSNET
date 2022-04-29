import scipy.io as sio
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix


def get_pair(pair_str):
    if pair_str == 'U_A':
        return 'A_U'
    elif pair_str == 'U_N':
        return 'N_U'
    elif pair_str == 'A_N':
        return 'N_A'
    else:
        return pair_str


def plot_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Generating normalized confusion matrix...")
    else:
        print('Generating confusion matrix without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
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


def plot_confusion_matrix(mat_path, class_names, save_prefix, acc):
    train_fn = sio.loadmat(mat_path)
    y_test = train_fn['target']
    y_pred = train_fn['pred']
    cnf_matrix = confusion_matrix(y_test.T, y_pred.T)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure(figsize=(16, 9))
    plot_matrix(cnf_matrix, classes=class_names, normalize=True,
                title='Normalized confusion matrix')
    save_path = os.path.join(save_prefix, 'Figures')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'test_{:.4f}.png'.format(acc)), format='png', dpi=2000)
    plt.close('all')
