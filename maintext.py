from model import get_model
from utils import levenshtein, check_and_retrieveVocabulary, data_preparation_CTC

from sklearn.model_selection import train_test_split
import cv2
import os
import sys
import numpy as np
import random
import itertools
import pickle
import tensorflow as tf
import argparse

CONST_IMG_DIR = "Data/PAGES/IMG/"
CONST_AGNOSTIC_DIR = "Data/PAGES/AGNOSTIC/"
PCKL_PATH = "Data/IAM_paragraph/"

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def load_data():
    X = []
    Y = []
    for file in os.listdir(CONST_IMG_DIR):
        X.append(cv2.imread(f"{CONST_IMG_DIR}{file}", 0))
        with open(f"{CONST_AGNOSTIC_DIR}{file.split('.')[0]}.agnostic") as agnosticfile:
            line = agnosticfile.readline()
            Y.append(['<sos>'] + line.split(" ") + ['<eos>'])
    
    return X, Y

def createDataArray(dataDict, folder):
    X = []
    Y = []
    for img in dataDict.keys():
        lines = dataDict[img]['lines']
        linearray = []
        for line in lines:
            line_stripped = line['text'].split(" ")
            for l in line_stripped:
                for char in l:
                    linearray += char
                linearray += ['<s>']
        Y.append(linearray)
        X.append(cv2.imread(f"{PCKL_PATH}/{folder}/{img}", 0))
    
    return X, Y

def load_data_text():
    trainX = []
    trainY = []
    valX = []
    valY = []
    testX = []
    testY = []

    with open(f"{PCKL_PATH}labels.pkl", "rb") as file:
        data = pickle.load(file)
        dataTrain = data['ground_truth']['train']
        dataVal = data['ground_truth']['valid']
        dataTest = data['ground_truth']['test']

        trainX, trainY = createDataArray(dataTrain, "train")
        valX, valY = createDataArray(dataVal, "valid")
        testX, testY = createDataArray(dataTest, "test")

    return trainX, trainY, valX, valY, testX, testY

def validateModel(model, X, Y, i2w):
    acc_ed_ser = 0
    acc_len_ser = 0

    randomindex = random.randint(0, len(X)-1)

    for i in range(len(X)):
        pred = model.predict(np.expand_dims(np.expand_dims(X[i],axis=0),axis=-1))[0]

        out_best = np.argmax(pred,axis=1)

        # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
        out_best = [k for k, g in itertools.groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c < len(i2w):  # CTC Blank must be ignored
                decoded.append(i2w[c])

        decoded.append('<eos>')
        groundtruth = [i2w[label] for label in Y[i]]

        if(i == randomindex):
            print(f"Prediction - {decoded}")
            print(f"True - {groundtruth}")

        acc_len_ser += len(Y[i])
        acc_ed_ser += levenshtein(decoded, groundtruth)


    ser = 100. * acc_ed_ser / acc_len_ser
    return ser

def parse_arguments():
    parser = argparse.ArgumentParser(description="Program arguments to work")
    parser.add_argument('--checkpoint', type=str, default=None, help="Model checkpoint to load")
    parser.add_argument('--save_path', type=str, help="Path to save the checkpoint")
    args = parser.parse_args()
    return args


def main():
    
    args = parse_arguments()

    XTrain, YTrain, XVal, YVal, XTest, YTest = load_data_text()

    #XTrain, XTest, YTrain, YTest = train_test_split(X,Y, test_size=0.1)
    w2i, i2w = check_and_retrieveVocabulary([YTrain, YVal, YTest], "./vocab", "SPAN")

    XTrain = np.array(XTrain)
    YTrain = np.array(YTrain)
    XVal = np.array(XVal)
    YVal = np.array(YVal)

    ratio = 150 / 300

    for i in range(len(XTrain)):
        img = (255. - XTrain[i]) / 255.
        width = int(np.ceil(img.shape[1] * ratio))
        height = int(np.ceil(img.shape[0] * ratio))
        XTrain[i] = cv2.resize(img, (width, height))
        for idx, symbol in enumerate(YTrain[i]):
            YTrain[i][idx] = w2i[symbol]
    for i in range(len(XVal)):
        img = (255. - XVal[i]) / 255.
        width = int(np.ceil(img.shape[1] * ratio))
        height = int(np.ceil(img.shape[0] * ratio))
        XVal[i] = cv2.resize(img, (width, height))
        for idx, symbol in enumerate(YVal[i]):
            YVal[i][idx] = w2i[symbol]

    model_train, model_pred = get_model(input_shape=(None, None, 1), out_tokens=len(w2i))
    
    if args.checkpoint != None:
        print(f"Loading checkpoint: {args.checkpoint}")
        model_train.load_weights(args.checkpoint)
    else:
        print(f"Saving checkpoint")
        model_train.save_weights("checkpoint.h5")
    
    X_train, Y_train, L_train, T_train = data_preparation_CTC(XTrain, YTrain, None)

    print('Training with ' + str(X_train.shape[0]) + ' samples.')
    
    inputs = {'the_input': X_train,
                 'the_labels': Y_train,
                 'input_length': L_train,
                 'label_length': T_train,
                 }
    
    outputs = {'ctc': np.zeros([len(X_train)])}
    
    best_ser = 10000
    not_improved = 0

    for super_epoch in range(10000):
       model_train.fit(inputs,outputs, batch_size = 2, epochs = 5, verbose = 2)
       CER = validateModel(model_pred, XVal, YVal, i2w)
       print(f"EPOCH {super_epoch} | CER {CER}")
       if CER < best_ser:
           print("CER improved - Saving epoch")
           model_train.save_weights(args.save_path)
           best_ser = CER
           not_improved = 0
       else:
           not_improved += 1
           if not_improved == 5:
               break

if __name__ == "__main__":
    main()