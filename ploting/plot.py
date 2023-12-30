import numpy as np
from matplotlib import pyplot as plt

import sys

sys.path.append('./')
from data_modules.unpickle import unpickle


def plot_image(array):
    
    red = array[:1024].reshape((32, 32))
    green = array[1024:2048].reshape((32, 32))
    blue = array[2048:].reshape((32, 32))
    
    rgb = np.dstack((red, green, blue))
    
    fig = plt.figure()
    
    plt.imshow(rgb)
    
    plt.show()
    
    
def plot_balance(labels):
    vals, counts = np.unique(labels, return_counts=True)
    vals = vals.astype(np.uint8)
    
    path_meta = './datasets/cifar/batches.meta'
    
    meta = unpickle(path_meta)
    label_names = meta[b'label_names']
    
    names = [label_names[i].decode("utf-8") for i in vals]
   
    fig, ax = plt.subplots()

    colors = ['red', 'brown', 'orange']
    
    ax.bar(names, counts, color=colors, label=names)

    ax.set_ylabel('Counts')
    ax.set_xlabel('Classes')
    ax.set_title('Class Balance')
    ax.legend(title='Targets')
    
    plt.show()