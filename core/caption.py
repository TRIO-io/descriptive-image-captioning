import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import string
import os
from PIL import Image
import glob
import pickle
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import backend as K

def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

from keras.models import model_from_json
def loadModel(model_file_name, weights_file_name = None):
    """Load keras model from disk."""
    if weights_file_name is None:
        weights_file_name = model_file_name
    # load json and create model
    json_file = open('model_weights/{}.json'.format(model_file_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_weights/{}.h5".format(weights_file_name))
    print("Loaded model {} from disk".format(model_file_name))
    return loaded_model

def greedySearch(photo, model, ixtoword, wordtoix):
    in_text = 'startseq'
    max_length = 34
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def return_captions(img_path, model_new):
    temp = preprocess(img_path)
    # x=plt.imread(img_path)
    # plt.imshow(x)
    # plt.show()
    fea_vec = model_new.predict(temp) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, (1,2048)) # reshape from (1, 2048) to (2048, )
    return fea_vec


def analyser(image_path):
    import pdb; pdb.set_trace()
    # load vocab
    ixtoword = load(open("pickle_files/ixtoword.pkl", "rb"))
    wordtoix = load(open("pickle_files/wordtoix.pkl", "rb"))

    # load image captioning model
    # model = loadModel("final_model")
    # model.load_weights('./model_weights/{}'.format("model_7.h5"))

    # loading Image net model 
    inc_model = InceptionV3(weights='imagenet')
    model_new = Model(inc_model.input, inc_model.layers[-2].output)

    # image_path = input()
    fea_vec = return_captions(image_path, model_new)
    caption = greedySearch(fea_vec, model_new, ixtoword, wordtoix)
    print("Caption: ",caption)
    return caption
