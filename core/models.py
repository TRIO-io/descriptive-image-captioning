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
from keras.models import model_from_json
from keras import backend as K

from django.db import models
from .caption import analyser
class Media(models.Model):
    caption = models.CharField(max_length=200, null=True, blank=True)
    media = models.FileField(null=True, blank=True)

    def __str__(self):
        return self.caption if self.caption else "Media - " + str(self.pk)
    
    def predict(self):
    
        K.reset_uids()
        
        ixtoword = load(open("pickle_files/ixtoword.pkl", "rb"))
        wordtoix = load(open("pickle_files/wordtoix.pkl", "rb"))
        
        inc_model = InceptionV3(weights='imagenet')
        model_new = Model(inc_model.input, inc_model.layers[-2].output)
        
        # model = loadModel("final_model")
        model_file_name = "final_model"
        weights_file_name = None
        if weights_file_name is None:
            weights_file_name = model_file_name
        # load json and create model
        json_file = open('model_weights/{}.json'.format(model_file_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model_weights/{}.h5".format(weights_file_name))
        model.load_weights('./model_weights/{}'.format("model_7.h5"))

        # Preprocess
        img = image.load_img(self.media, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Imagenet Feature Vector
        fea_vec = model_new.predict(x) 
        fea_vec = np.reshape(fea_vec, (1,2048))

        # Getting Caption
        in_text = 'startseq'
        max_length = 34
        for i in range(max_length):
            sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([fea_vec,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final

        
    #     model = 'cnn/model/model_mobilenet.json'
    #     weights = 'cnn/model/weights_mobilenet.h5'
    #     with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    #         with open(model, 'r') as f:
    #             model = model_from_json(f.read())
    #             model.load_weights(weights)
    #     img = image.load_img(self.img, target_size=(224,224))
    #     x = image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0) 
    #     x = preprocess_input(x)
    #     result = model.predict(x)
    #     result_decode = imagenet_utils.decode_predictions(result)
    #     for (i, (predId, pred, prob)) in         enumerate(result_decode[0]):
    #         return "{}.-  {}: {:.2f}%".format(i + 1, pred, prob * 100)

    def save(self, *args, **kwargs):
        self.caption = self.predict()
        super().save(*args, **kwargs)