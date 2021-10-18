from model import get_line_model
from utils import ctc_batch_generator, check_and_retrieveVocabulary, levenshtein

from sklearn.model_selection import train_test_split
import cv2
import os
import sys
import numpy as np
import random
import itertools
import tqdm
import tensorflow as tf
import argparse

CONST_DIR = "Data/PRIMUS/package_aa/"
BATCH_SIZE = 16

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def load_data():
    X = []
    Y = []
    i = 0
    for folder in tqdm.tqdm(os.listdir(CONST_DIR)):
        with open(f"{CONST_DIR}{folder}/{folder}.agnostic") as agnostic:
            line = agnostic.readline()
            Y.append([token for token in line.split("\t")])
        
        X.append(cv2.imread(f"{CONST_DIR}{folder}/{folder}.png", 0))
        if i == 7000:
            break
        i+=1

    return X, Y

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

    X, Y= load_data()

    #XTrain, XTest, YTrain, YTest = train_test_split(X,Y, test_size=0.1)
    w2i, i2w = check_and_retrieveVocabulary([Y], "./vocab", "SPANStaves")

    XTrain, XVal, YTrain, YVal = train_test_split(X, Y, test_size=0.25)

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
       model_train.fit(batch_generator, steps_per_epoch=len(XTrain)//BATCH_SIZE, epochs = 1, verbose = 1)
       SER = validateModel(model_pred, XVal, YVal, i2w)
       print(f"EPOCH {super_epoch} | SER {SER}")
       if SER < best_ser:
           print("SER improved - Saving epoch")
           model_train.save_weights(args.save_path)
           model_base.save_weights("models/SPAN_LINES_PRET.h5")
           best_ser = SER

if __name__ == "__main__":
    main()