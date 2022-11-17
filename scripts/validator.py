import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score,f1_score, accuracy_score

def compute_confusion_matrix(observed, predicted):
   cf = confusion_matrix(observed, predicted).astype(float)
   cf = cf / cf.sum(axis=1)[:, np.newaxis]
   return cf.diagonal()

def compute_accuracy(observed, predicted):
    return accuracy_score(observed, predicted)

def compute_kappa(observed, predicted):
    return cohen_kappa_score(observed, predicted)

