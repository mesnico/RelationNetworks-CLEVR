import itertools
import numpy as np
import pickle
import argparse
import matplotlib
import os

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes, cmap,
                          normalize=False,
                          title='Confusion matrix'):
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

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    for axis in [ax.xaxis, ax.yaxis]:
      axis.set_ticks(tick_marks+0.5, minor=True)
      axis.set(ticks=tick_marks, ticklabels=classes)

    labels = ax.get_xticklabels()
    for label in labels:
        label.set_rotation(45)
    #plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.grid(True, which='minor')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot RN confusion matrix')
  parser.add_argument('file', type=str, help='Stat file to use for plotting')
  parser.add_argument('--no-show', action='store_true', help='Do not show plot, store only on file')
  args = parser.parse_args()

  img_dir = 'imgs/'
  args.img_dir = img_dir

  if not os.path.exists(img_dir):
    os.makedirs(img_dir)

  if args.no_show:
      matplotlib.use('Agg')    
  import matplotlib.pyplot as plt

  # Load stats file
  filename = open(args.file, 'rb')
  p = pickle.load(filename)
  target = p['confusion_matrix_target']
  pred = p['confusion_matrix_pred']
  class_names = p['confusion_matrix_labels']

  # Compute confusion matrix
  cnf_matrix = confusion_matrix(target, pred)
  np.set_printoptions(precision=2)

  # Plot normalized confusion matrix
  plot_confusion_matrix(cnf_matrix, classes=class_names, cmap=plt.cm.Blues, normalize=True,
                        title='Normalized confusion matrix')

  plt.savefig(os.path.join(args.img_dir, 'confusion.png'))
  if not args.no_show:
    plt.show()
