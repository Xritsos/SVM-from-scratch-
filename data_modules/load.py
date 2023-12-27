import numpy as np
import sys

sys.path.append('./')
from data_modules import unpickle


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


def preprocess(y_train, y_test, class_id):
    # the targer class each time will be assigned to 1
    y_train[y_train==class_id] = 1
    y_train[y_train!=1] = -1
    
    y_train = y_train.astype(np.float64)
    
    print()
    print(f"Train labels: {np.unique(y_train)}")
    
    y_test[y_test==class_id] = 1
    y_test[y_test!=1] = -1
    
    y_test = y_test.astype(np.float64)
    
    print()
    print(f"Test labels: {np.unique(y_test)}")
    
    
    return y_train, y_test


def subsample(x_train, y_train, class_id):
    classes = [3, 4, 7]
    classes.remove(class_id)
    
    target_tr = y_train==class_id
    # target_te = y_test==class_id
    
    max_sample_tr = math.floor(x_train[target_tr, :].shape[0] / 2)
    # max_sample_te = math.floor(x_test[target_te, :].shape[0] / 2)
    
    other_1_tr = y_train==classes[0]
    other_2_tr = y_train==classes[1]
    
    # other_1_te = y_test==classes[0]
    # other_2_te = y_test==classes[1]
    
    x_train_class = x_train[target_tr, :]
    # x_test_class = x_test[target_te, :]
    y_train_class = y_train[target_tr]
    # y_test_class = y_test[target_te]
    
    # train
    x_train_other_1 = x_train[other_1_tr, :]
    y_train_other_1 = y_train[other_1_tr]
    
    rand = np.random.randint(0, x_train_other_1.shape[0], max_sample_tr)
    
    x_train_other_1 = x_train_other_1[rand, :]
    y_train_other_1 = y_train_other_1[rand]
    
    x_train_other_2 = x_train[other_2_tr, :]
    y_train_other_2 = y_train[other_2_tr]
    
    rand = np.random.randint(0, x_train_other_2.shape[0], max_sample_tr)
    
    x_train_other_2 = x_train_other_2[rand, :]
    y_train_other_2 = y_train_other_2[rand]
    
    x_train_new = np.concatenate((x_train_class,
                                  x_train_other_1,
                                  x_train_other_2))
    
    y_train_new = np.concatenate((y_train_class,
                                  y_train_other_1,
                                  y_train_other_2))
    
    # test
    # x_test_other_1 = x_test[other_1_te, :]
    # y_test_other_1 = y_test[other_1_te]
    
    # rand = np.random.randint(0, x_test_other_1.shape[0], max_sample_te)
    
    # x_test_other_1 = x_test_other_1[rand, :]
    # y_test_other_1 = y_test_other_1[rand]
    
    # x_test_other_2 = x_test[other_2_te, :]
    # y_test_other_2 = y_test[other_2_te]
    
    # rand = np.random.randint(0, x_test_other_2.shape[0], max_sample_te)
    
    # x_test_other_2 = x_test_other_2[rand, :]
    # y_test_other_2 = y_test_other_2[rand]
    
    # x_test_new = np.concatenate((x_test_class,
    #                               x_test_other_1,
    #                               x_test_other_2))
    
    # y_test_new = np.concatenate((y_test_class,
    #                               y_test_other_1,
    #                               y_test_other_2))
    
    return x_train_new, y_train_new