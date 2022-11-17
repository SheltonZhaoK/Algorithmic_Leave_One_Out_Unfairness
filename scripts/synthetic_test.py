import random, tensorflow, numpy, os, pickle, time
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from numpy.random import seed
from statistics import mean

from data import *
from models import *
from validator import *

def plot_training_loss(history, numEpochs):
    loss_train = history.history['loss']
    #print(loss_train)
    categorical_accuracy = history.history['categorical_accuracy']
    #print("loss:",loss_train)
    #print("categorical_accuracy",categorical_accuracy)
    epochs = range(0,numEpochs)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, categorical_accuracy, 'b', label='Categorical accuracy')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('model_training_convergence_512_1.jpg')
    #plt.show()  
    
def set_seed(seed):
    random.seed(seed)
    tensorflow.random.set_seed(seed)
    numpy.random.seed(seed)

def read_data(fileName, labelName, label2predict):
    data = pd.read_csv(fileName, index_col = 0)
    labels = pd.read_csv(labelName, "\t")
    data[label2predict] = labels[label2predict].tolist()
    return data

def prepare_data(annotation, label2predict, fraction):
    '''
    The format of annotation must be: 
        The instances name are index.
        Labels to be predicted are in the last column, where column index should be equal to label2predict.
    '''
    data = Data(annotation, label2predict, fraction)
    data.set_label()
    data_balanced = data.balance_data(annotation, label2predict)
    data.annotation = data_balanced
    data.split_data()
    return data

def evaluate_ffnn(ffnn_accuracy, ffnn_kappa, ffnn_predicted, ffnn_confidence_level, LUF, train_x, train_y, train_cellId, \
test_x, test_y, test_cellId, num_labels, remaining_iter, model_saved):
    index = train_cellId.index(LUF)
    new_train_x, new_train_y = np.delete(train_x, index ,0), np.delete(train_y, index ,0)
    ffnn_model, _ = train_ffnn_model(new_train_x, new_train_y, num_labels, remaining_iter, model_saved)
    LUF_predicted, LUF_confidence_level, LUF_observed = test_ffnn_model(ffnn_model, test_x, test_y)

    changeOfAccuracy = abs(ffnn_accuracy - compute_accuracy(LUF_observed, LUF_predicted))
    changeOfKappa = abs(ffnn_kappa - compute_kappa(LUF_observed, LUF_predicted))
    numPointsFlipped = np.count_nonzero(ffnn_predicted - LUF_predicted)/len(test_cellId)
    changeOfConfidenceLevel = np.abs(ffnn_confidence_level - LUF_confidence_level).max()
    return [changeOfAccuracy, changeOfKappa, numPointsFlipped, changeOfConfidenceLevel]

def evaluate_lr(lr_accuracy, lr_kappa, lr_predicted, lr_confidence_level, LUF, train_x, train_y, train_cellId, \
test_x, test_y, test_cellId, remaining_iter, model_saved):
    index = train_cellId.index(LUF)
    new_train_x, new_train_y = np.delete(train_x, index ,0), np.delete(train_y, index ,0)

    lr_model = train_lr_model(new_train_x, new_train_y, remaining_iter, model_saved)
    LUF_predicted, LUF_confidence_level, LUF_observed = test_lr_model(lr_model, test_x, test_y)

    changeOfAccuracy = abs(lr_accuracy - compute_accuracy(LUF_observed, LUF_predicted))
    changeOfKappa = abs(lr_kappa - compute_kappa(LUF_observed, LUF_predicted))
    numPointsFlipped = np.count_nonzero(lr_predicted - LUF_predicted)/len(test_cellId)
    changeOfConfidenceLevel = np.abs(lr_confidence_level - LUF_confidence_level).max()
    return [changeOfAccuracy, changeOfKappa, numPointsFlipped, changeOfConfidenceLevel]

def main(fileDir, label2predict, fraction, LUFSize):
    #set_seed(1)
    pool = mp.Pool(mp.cpu_count())
    initial_iter, remaining_iter = 2, 100
    ffnnModelPath = "./ffnnModel.h5"
    
    results = pd.DataFrame(columns = ['Algorithms','Total Instances', 'num_classes' ,'Training #',\
         'Testing #','accuracy', 'kappa','ch_accuracy','ch_kappa', \
         'points_flipped', 'ave_ch_confi', 'max_ch_confi'])

    sizes = [i for i in range(1, 11)]
    num_classes = [i for i in range(1, 7)]
    #sizes = [i for i in range(1, 3)]
    #num_classes = [i for i in range(1, 3)]

    for size in sizes:
        for num_class in num_classes:

            '''
            Prepare training and testing data
            '''
            tempDir = fileDir + "data_" + str(size) + "k/"
            annotation = read_data(tempDir + 'synthetic_embeddings_' + str(num_class) + '.csv' , \
            tempDir + 'labels_' + str(num_class) + '.txt', label2predict)
            
            data = prepare_data(annotation, label2predict, fraction)
            train_x, train_y, train_cellId, num_labels, = data.make_data("training")
            test_x, test_y, test_cellId, _ = data.make_data("testing")

            LUF_id = data.make_LUFData(LUFSize)

            '''
            Pre-train ffnn_model and finish training
            '''
            ffnn_model, history = train_ffnn_model(train_x, train_y, num_labels, initial_iter, None)
            ffnn_model.save(ffnnModelPath)
            ffnn_model, history = train_ffnn_model(train_x, train_y, num_labels, remaining_iter, ffnnModelPath)  
            #plot_training_loss(history, 50) #fixme          
            ffnn_predicted, ffnn_confidence_level, observed = test_ffnn_model(ffnn_model, test_x, test_y)
            ffnn_accuracy = compute_accuracy(observed, ffnn_predicted)
            ffnn_kappa = compute_kappa(observed, ffnn_predicted)

            '''
            Pre-train lr model and finish training
            '''
            lr_model = train_lr_model(train_x, train_y, initial_iter, None)
            lr_save = pickle.dumps(lr_model)
            lr_model = train_lr_model(train_x, train_y, remaining_iter, lr_save)
            lr_predicted, lr_confidence_level, lr_observed = test_lr_model(lr_model, test_x, test_y)
            lr_accuracy = compute_accuracy(observed, lr_predicted)
            lr_kappa = compute_kappa(observed, lr_predicted)
            
            '''
            Begin parallel ffnn & lr LUF experiment
            '''
            ffnn_parallelObjects = [pool.apply_async(evaluate_ffnn, args = (ffnn_accuracy, ffnn_kappa, ffnn_predicted, \
                                ffnn_confidence_level, LUF, train_x, train_y, train_cellId,test_x, test_y, \
                                test_cellId, num_labels, remaining_iter, ffnnModelPath)) for LUF in LUF_id]
            ffnn_results = [objects.get() for objects in ffnn_parallelObjects]
            df = pd.DataFrame(ffnn_results)
            tempt = ["FFNN", size*1000, 2**num_class, len(train_y), len(test_y), ffnn_accuracy, ffnn_kappa]
            tempt.extend(df.mean().to_list())
            tempt.append(max(df.iloc[:,-1].to_list()))
            results.loc[len(results)] = tempt

            lr_parallelObjects = [pool.apply_async(evaluate_lr, args = (lr_accuracy, lr_kappa, lr_predicted, \
                                lr_confidence_level, LUF, train_x, train_y, train_cellId, test_x, test_y, \
                                test_cellId, remaining_iter, lr_save)) for LUF in LUF_id]
            lr_results = [objects.get() for objects in lr_parallelObjects]
            df = pd.DataFrame(lr_results)
            tempt = ["Logistic Regression", size*1000, 2**num_class, len(train_y), len(test_y), lr_accuracy, lr_kappa]
            tempt.extend(df.mean().to_list())
            tempt.append(max(df.iloc[:,-1].to_list()))
            results.loc[len(results)] = tempt
            
    results.iloc[:,4:] = results.iloc[:,4:].round(decimals = 3)
    print(results)
    results.to_csv('../output/experiment_ffnn_lr_100.csv')
    os.remove(ffnnModelPath)
    
if __name__ == '__main__':
    fileDir = '../data/synthetic_data/'
    label2predict = 'labels'
    fraction = 0.2 #1/5
    LUFSize = 100
    main(fileDir, label2predict, fraction, LUFSize)
