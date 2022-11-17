'''
Supervised Machine Learning Models for Classification
'''
import pickle

import xgboost as xgb
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation,Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2

'''
Feed Forward Neural Network
'''
def train_ffnn_model(train_x, train_y, num_labels, numIter, saved_model):
   if(saved_model == None):
      num_layers = 1
      num_nodes = 512
      #num_nodes = 128
      #num_layers = 2

      model = Sequential()
      model.add(Dense(num_nodes, input_dim=train_x.shape[1], \
               kernel_regularizer=l2(0.01), activation="relu"))
      for l in range(1, num_layers):
         num_nodes = int(num_nodes / 2)
         model.add(Dense(num_nodes, activation="relu"))
      model.add(Dense(num_labels, activation="softmax"))

      model.compile(loss='categorical_crossentropy', \
                     optimizer=optimizers.RMSprop(learning_rate = 1e-4), \
                     metrics=['categorical_accuracy'])
      history = model.fit(train_x, train_y, epochs=numIter, batch_size=128,verbose=0)
   else:
      model = load_model(saved_model)
      history = model.fit(train_x, train_y, epochs=numIter, batch_size=128,verbose=0)

   return(model, history)
   
def test_ffnn_model(model, test_data, test_labels):
   confidence_level = model.predict(test_data)
   predicted_labels = np.argmax(confidence_level, axis=-1)
   lables = np.argmax(test_labels, axis=-1)
   return predicted_labels, confidence_level, lables

def train_lr_model(train_x, train_y, numIter, saved_model):
   if saved_model == None:
      train_y = np.argmax(train_y, axis=-1)
      model = LogisticRegression(verbose = 0, max_iter = numIter, warm_start = True)
      model.fit(train_x, train_y)
   else:
      train_y = np.argmax(train_y, axis=-1)
      model = pickle.loads(saved_model)
      model.fit(train_x, train_y)
   return model

def test_lr_model(model, test_data, test_labels):
   confidence_level = model.predict_proba(test_data)
   predicted_labels = np.argmax(confidence_level, axis=-1)
   lables = np.argmax(test_labels, axis=-1)
   return predicted_labels, confidence_level, lables

'''
Extreme Gradient Boosting Model
'''
def train_xgb_model(train_x, train_y, num_labels):

    d_train = xgb.DMatrix(data=train_x, label=np.argmax(train_y, axis= -1))

    param = {'max_depth': 4, 'eta': 1, 'objective': 'multi:softprob', 'num_class': num_labels}
    param['nthread'] = 4
    param['eval_metric'] = 'mlogloss'
    num_round = 50

    model  = xgb.train(param, d_train, num_round)
    return model

def test_xgb_model(model, test_data, test_labels, xgb_result,accuracy_xgb,kappa_xgb):
   d_test = xgb.DMatrix(test_data)
   predicted = np.argmax(model.predict(d_test), axis=-1)
   observed = np.argmax(test_labels, axis=-1)
   accuracy_xgb.append(accuracy_score(observed, predicted))
   kappa_xgb.append(cohen_kappa_score(observed, predicted))
   xgb_result.loc[len(xgb_result)] = evaluate_performance(observed, predicted)
   return

'''
K-nearest neibors model
'''
def train_knn_model(train_x, train_y, num_labels):
    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(train_x, train_y)

    return model 

def test_knn_model(model, test_data, test_labels, knn_result, accuracy_knn, kappa_knn):
   predicted = np.argmax(model.predict(test_data), axis=-1)
   observed = np.argmax(test_labels, axis=-1)
   accuracy_knn.append(accuracy_score(observed, predicted))
   kappa_knn.append(cohen_kappa_score(observed, predicted))
   knn_result.loc[len(knn_result)] = evaluate_performance(observed, predicted)
   return

