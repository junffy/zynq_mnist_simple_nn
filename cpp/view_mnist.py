# coding: utf-8
def view_mnist(first_offset, last_offset):
  import sys, os
  sys.path.append(os.pardir)

  import numpy as np
  from dataset.mnist import load_mnist
  import matplotlib.pyplot as plt

  # データの読み込み
  (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=True)

  fig = plt.figure()
  fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)

  current_view = 1
  for i in range(first_offset, last_offset):
    ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
    ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    current_view += 1

  plt.show()
view_mnist(0, 10)
