
import numpy as np
import random


def train_test_split(X, y, indices_cv, train_prop, seed):
    """
    Return X_train, X_test, y_train and y_test arrays.

    Assures that the X_test and y_test contain different 
    labels and that the test set contains at least 3 pilots.
    """

    # Initialize arrays at 0
    X_train = np.zeros_like(X)
    X_test = np.zeros_like(X)
    y_train = np.zeros_like(y)
    y_test = np.zeros_like(y)

    seed_increment = 0
    while (np.mean(y_test) == 1) \
            or (np.mean(y_test) == 0) \
            or (np.unique(indices_cv[testing_indices, 1]).size < 3):

        # Cross-validation repartition
        training_indices, testing_indices = TimeCV(
            indices_cv, 1-train_prop, weighting=False,
            seed_k=seed+seed_increment
        )
        X_train = X[training_indices, :]
        X_test = X[testing_indices, :]
        y_train = y[training_indices]
        y_test = y[testing_indices]

        # Increment
        seed_increment += 1

    pilots_tested = np.unique(indices_cv[testing_indices, 1])

    return X_train, X_test, y_train, y_test, pilots_tested, testing_indices


def TimeCV(T, per_test, weighting, seed_k):
    """ 
    T: is a two-column vector where COLUMN0 indicates the number of the task in time order, and 
    COLUMN1 indicates the individual.
    per_test: is the percentage (from 0 to 1) of test set

    Created on Fri Aug  26 11:40:19 2022
    @author: ibargiotas
    """
    # np.random.seed(seed_k)
    per_temp = 0
    ID = (T[:, 0]*0 > 1)  # A vector full of false's
    while per_temp <= per_test:
        #idx = np.random.randint(0, ID.shape[0], size=1)
        idx = random.choice(np.arange(0, ID.shape[0]))
        full_idx = np.all(
            [T[:, 0] >= T[idx, 0], T[:, 1] == T[idx, 1]], axis=0)
        ID[np.where(full_idx)[0]] = True

        per_temp = np.count_nonzero(ID)/ID.shape[0]

    test_id = np.where(ID)[0]
    train_id = np.where(~ID)[0]

    if weighting == True:
        U = np.unique(T[test_id, 1])
        for i in U:
            x_tmp = train_id[T[train_id, 1] == i]
            train_id = np.append(train_id, x_tmp)
            train_id = np.append(train_id, x_tmp)

    return train_id, test_id
