from model import get_line_model
from utils import ctc_batch_generator, check_and_retrieveVocabulary

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
import editdistance

CONST_IMG_DIR = "Data/PAGES/IMG/"
CONST_AGNOSTIC_DIR = "Data/PAGES/AGNOSTIC/"
PCKL_PATH = "Data/IAM_lines/"
BATCH_SIZE = 16

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def edit_cer_from_list(truth, pred):
    edit = 0
    for t, p in zip(truth, pred):
        edit += editdistance.eval(t, p)
    return edit


def edit_wer_from_list(truth, pred):
    edit = 0
    separation_marks = ["?", ".", ";", ",", "!", "\n"]
    for pred, gt in zip(pred, truth):
        for mark in separation_marks:
            gt.replace(mark, " {} ".format(mark))
            pred.replace(mark, " {} ".format(mark))
        gt = gt.split(" ")
        pred = pred.split(" ")
        while '' in gt:
            gt.remove('')
        while '' in pred:
            pred.remove('')
        edit += editdistance.eval(gt, pred)
    return edit

def nb_words_from_list(list_gt):
    separation_marks = ["?", ".", ";", ",", "!", "\n"]
    len_ = 0
    for gt in list_gt:
        for mark in separation_marks:
            gt.replace(mark, " {} ".format(mark))
        gt = gt.split(" ")
        while '' in gt:
            gt.remove('')
        len_ += len(gt)
    return len_


def createDataArray(dataDict, folder):
    X = []
    Y = []
    for sample in dataDict.keys():
        if type(dataDict[sample]) == str:
            Y.append([char for char in dataDict[sample]])
        else:
            Y.append([char for char in dataDict[sample]['text']])

        X.append(cv2.imread(f"{PCKL_PATH}/{folder}/{sample}", 0))
    
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
    randomindex = random.randint(0, len(X)-1)
    acc_cer = 0
    acc_wer = 0
    for i in range(len(X)):
        pred = model.predict(np.expand_dims(np.expand_dims(X[i],axis=0),axis=-1))[0]

        out_best = np.argmax(pred,axis=1)

        # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
        out_best = [k for k, g in itertools.groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c < len(i2w):  # CTC Blank must be ignored
                decoded.append(i2w[c])

        groundtruth = [i2w[label] for label in Y[i]]

        if(i == randomindex):
            print(f"Prediction - {decoded}")
            print(f"True - {groundtruth}")

        characters = len("".join(groundtruth))
        words = nb_words_from_list(["".join(groundtruth)])

        ed_cer = edit_cer_from_list(["".join(decoded)], ["".join(groundtruth)])


        ed_wer = edit_wer_from_list(["".join(decoded)], ["".join(groundtruth)])


        acc_cer += ed_cer / characters
        acc_wer += ed_wer / words
    
    cer = 100.*acc_cer / len(X)
    wer = 100.*acc_wer / len(X)
    
    return cer, wer

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
    w2i, i2w = check_and_retrieveVocabulary([YTrain, YVal, YTest], "./vocab", "SPANLines")

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

    model_train, model_pred, model_base = get_line_model(input_shape=(None, None, 1), out_tokens=len(w2i))
    
    if args.checkpoint != None:
        print(f"Loading checkpoint: {args.checkpoint}")
        model_train.load_weights(args.checkpoint)

    print('Training with ' + str(XTrain.shape[0]) + ' samples.')
    
    best_ser = 10000

    batch_generator = ctc_batch_generator(BATCH_SIZE, XTrain, YTrain, True)

    for super_epoch in range(5000):
       model_train.fit(batch_generator, steps_per_epoch=len(XTrain)//BATCH_SIZE, epochs = 1, verbose = 2)
       CER, WER = validateModel(model_pred, XVal, YVal, i2w)
       print(f"EPOCH {super_epoch} | CER {CER} | WER {WER}")
       if CER < best_ser:
           print("CER improved - Saving epoch")
           model_train.save_weights(args.save_path)
           model_base.save_weights("models/SPAN_LINES_PRET.h5")
           best_ser = CER

if __name__ == "__main__":
    main()