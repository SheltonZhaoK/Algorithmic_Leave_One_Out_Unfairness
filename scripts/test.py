import random, tensorflow, numpy, os
import pandas as pd
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
    plt.savefig('model_training_convergence.jpg')
    #plt.show()  
    
def set_seed(seed):
    random.seed(seed)
    tensorflow.random.set_seed(seed)
    numpy.random.seed(seed)

def read_data(fileName, labelName, label2predict, label2balance):
    data = pd.read_csv(fileName, index_col = 0)
    labels = pd.read_csv(labelName)
    data[label2predict], data[label2balance] = labels[label2predict].tolist(), labels[label2balance].tolist()   
    return data

def prepare_data(annotation, label2predict, label2balance, fraction):
    '''
    The format of annotation must be: 
        The instances name are index.
        Labels to be predicted are in the last column, where column index should be equal to label2predict.
    '''
    data = Data(annotation, label2predict, fraction)
    data.set_label()
    data.balance_data()
    data.split_data()
    return data

def main(fileName, labelName, label2predict, label2balance, fraction, LUFSize):
    set_seed(1)
    initial_epoch, remaining_epoch = 2, 50
    modelPath = "./ffnnModel.h5"
    
    results = pd.DataFrame(columns = ['Instance #', 'Training #',\
         'Testing #','accuracy', 'kappa','ch_accuracy','ch_kappa', 'points_flipped', 'ave_ch_confi', 'max_ch_confi'])

    for size in [1000, 5000, 10000, 20000,40000, 60000, 80000,100000,150000,200000]:
        #for size in [1000,5000]:
        annotation = read_data(fileName, labelName, label2predict, label2balance)
        annotation = annotation.sample(n = size)
        data = prepare_data(annotation, label2predict, label2balance, fraction)

        train_x, train_y, train_cellId, num_labels, = data.make_data("training")
        test_x, test_y, test_cellId, _ = data.make_data("testing")

        ffnn_model, history = train_ffnn_model(train_x, train_y, num_labels, initial_epoch, None)
        #plot_training_loss(history, 50)

        ffnn_model.save(modelPath)
        ffnn_model, history = train_ffnn_model(train_x, train_y, num_labels, remaining_epoch, modelPath)
        

        predicted, confidence_level, observed = test_ffnn_model(ffnn_model, test_x, test_y)
        accuracy = compute_accuracy(observed, predicted)
        kappa = compute_kappa(observed, predicted)
        print("accuracy: %.3f\nkappa: %.3f"%(accuracy, kappa))
        LUF_id = data.make_LUFData(LUFSize)

        changeOfAccuracy = []
        changeOfKappa = []
        changeOfConfidenceLevel = []
        pointsFlipped = []

        i = 1
        for ind in LUF_id:
            index = train_cellId.index(ind)
            new_train_x, new_train_y = np.delete(train_x, index ,0), np.delete(train_y, index ,0)
            ffnn_model, _ = train_ffnn_model(new_train_x, new_train_y, num_labels, remaining_epoch, modelPath)
            LUF_predicted, LUF_confidence_level, LUF_observed = test_ffnn_model(ffnn_model, test_x, test_y)
            
            changeOfAccuracy.append(abs(accuracy - compute_accuracy(LUF_observed, LUF_predicted)))
            changeOfKappa.append(abs(kappa - compute_kappa(LUF_observed, LUF_predicted)))
            pointsFlipped.append(np.count_nonzero(predicted - LUF_predicted)/len(test_cellId))
            changeOfConfidenceLevel.append(np.abs(confidence_level - LUF_confidence_level).max())
            
            print("%d LUF is done ===========" % (i))
            i += 1
            
        results.loc[len(results)] = [size, len(train_cellId), len(test_cellId), "{:.3f}".format(accuracy), \
        "{:.3f}".format(kappa), "{:.3f}".format(mean(changeOfAccuracy)), "{:.3f}".format(mean(changeOfKappa)),"{:.3f}".format(mean(pointsFlipped)),\
        "{:.3f}".format(mean(changeOfConfidenceLevel)),"{:.3f}".format(max(changeOfConfidenceLevel))]
    
    print(results)
    results.to_csv('/deac/csc/khuriGrp/zhaok220/ml_fairness/output/size_dependency_512_1_harmonized_umap.csv')
    os.remove(modelPath)

if __name__ == '__main__':
    fileName = '/deac/csc/khuriGrp/khurin/nathan/data/kidney/harmonized_umap.csv'
    #fileName = '/deac/csc/khuriGrp/khurin/nathan/data/kidney/harmonized.csv'
    labelName = '/deac/csc/khuriGrp/khurin/nathan/data/kidney/labels.csv'
    label2predict = 'STATUS'
    label2balance = 'DID'
    fraction = 0.2 #1/5
    LUFSize = 100
    main(fileName, labelName, label2predict, label2balance, fraction, LUFSize)
