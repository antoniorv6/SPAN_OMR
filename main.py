from model import get_model
from utils import levenshtein, check_and_retrieveVocabulary, data_preparation_CTC

from sklearn.model_selection import train_test_split
import cv2
import os
import sys
import numpy as np

CONST_IMG_DIR = "Data/PAGES/IMG/"
CONST_AGNOSTIC_DIR = "Data/PAGES/AGNOSTIC/"

fixed_height = 512

def load_data():
    X = []
    Y = []
    for file in os.listdir(CONST_IMG_DIR):
        X.append(cv2.imread(f"{CONST_IMG_DIR}{file}", 0))
        with open(f"{CONST_AGNOSTIC_DIR}{file.split('.')[0]}.agnostic") as agnosticfile:
            line = agnosticfile.readline()
            Y.append(line.split(" "))
    
    return X, Y


def main():
    X, Y = load_data()
    print(X[0])
    print(Y[0])

    XTrain, XTest, YTrain, YTest = train_test_split(X,Y, test_size=0.1)
    w2i, i2w = check_and_retrieveVocabulary([YTrain, YTest], "./vocab", "SPAN")

    XTrain = np.array(XTrain)
    YTrain = np.array(YTrain)
    XTest = np.array(XTest)
    YTest = np.array(YTest)

    for i in range(len(XTrain)):
        img = (255. - XTrain[i]) / 255.
        width = fixed_height
        XTrain[i] = cv2.resize(img, (width, fixed_height))
        for idx, symbol in enumerate(YTrain[i]):
            YTrain[i][idx] = w2i[symbol]
    
    for i in range(len(XTest)):
        img = (255. - XTest[i]) / 255.
        width = fixed_height
        XTest[i] = cv2.resize(img, (width, fixed_height))
        for idx, symbol in enumerate(YTest[i]):
            YTest[i][idx] = w2i[symbol]

    print(XTrain.shape)
    print(YTrain.shape)

    model_train, model_pred = get_model(input_shape=(fixed_height,fixed_height,1), out_tokens=256)

    X_train, Y_train, L_train, T_train = data_preparation_CTC(XTrain, YTrain, fixed_height)

    print('Training with ' + str(X_train.shape[0]) + ' samples.')
    
    inputs = {'the_input': X_train,
                 'the_labels': Y_train,
                 'input_length': L_train,
                 'label_length': T_train,
                 }
    
    outputs = {'ctc': np.zeros([len(X_train)])}
    best_ser = 10000

    for super_epoch in range(10000):
       model_train.fit(inputs,outputs, batch_size = 16, epochs = 5, verbose = 2)

    pass

if __name__ == "__main__":
    main()