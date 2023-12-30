import numpy as np
import math
import sys

sys.path.append('./')
from data_modules.unpickle import unpickle


def get_data(path):
    
    batch = unpickle(path)
    
    data = batch[b'data']
    labels = np.asarray(batch[b'labels'])

    # we choose only the classes deer horse and cat
    # deer --> 4, horse --> 7, cat --> 3
    
    deer = data[labels==4, :]
    horse = data[labels==7, :]
    cat = data[labels==3, :]
    
    new_data = np.concatenate((deer, horse, cat))
    
    deer_lb = labels[labels==4]
    horse_lb = labels[labels==7]
    cat_lb = labels[labels==3]
    
    new_labels = np.concatenate((deer_lb, horse_lb, cat_lb))
    
    return new_data, new_labels


def preprocess(y, class_id):
    # the targer class each time will be assigned to 1
    y[y==class_id] = 1
    y[y!=1] = -1
    
    y = y.astype(np.float64)
    
    return y


def subsample(x, y, class_id):
    classes = [3, 4, 7]
    classes.remove(class_id)
    
    target = y==class_id
    
    max_sample = math.floor(x[target, :].shape[0] / 2)
    
    other_1 = y==classes[0]
    other_2 = y==classes[1]
    
    x_class = x[target, :]
    y_class = y[target]
    
    x_other_1 = x[other_1, :]
    y_other_1 = y[other_1]
    
    x_other_1 = x_other_1[:max_sample, :]
    y_other_1 = y_other_1[:max_sample]
    
    x_other_2 = x[other_2, :]
    y_other_2 = y[other_2]
    
    x_other_2 = x_other_2[:max_sample, :]
    y_other_2 = y_other_2[:max_sample]
    
    x_new = np.concatenate((x_class, x_other_1, x_other_2))
    
    y_new = np.concatenate((y_class, y_other_1, y_other_2))
    
    return x_new, y_new
