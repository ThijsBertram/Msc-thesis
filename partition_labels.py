import os
import re
import shutil
from shutil import copyfile
from PIL import Image
import numpy as np
import pickle
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K

# Change directories if replicating
train_dir = 'D:/Rijksmuseum/train/'
test_dir = 'D:/Rijksmuseum/test/'
val_dir = 'D:/Rijksmuseum/val/'
xml_dir = 'D:/Rijksmuseum/xml/'
data_dir = 'D:/Rijksmuseum/data2/'

folders = [str(i) for i in range(0, 34)]
# dictionary that will store label info (corresponding artist)
artists = dict()
# dictionary that will store train, test and val labels
labels = dict()
# dictionary that stores IDs for train, test and val splits
partition = dict()
# create empty lists that will be filled with IDs
partition['train'] = []
partition['test'] = []
partition['val'] = []

# fill artists dictionary
for el in folders:
    train_file = os.listdir(train_dir + el)[0]
    train_file = train_file[:train_file.find('.jpg')] + '.xml'

    with open(xml_dir + train_file, 'r', encoding='utf-8') as f:
        text = f.read()
        m = re.search('(?<=<dc:creator>)(.*)(?=<\/dc:creator>)', text).group(0).strip()
        train_artist = re.search('(?<=:)(.*)', m).group(0).strip()
    artists['{}'.format(el)] = train_artist

# fill labels and partition dictionarys
for i, el in enumerate(folders):
    # train
    files = os.listdir(train_dir + el)
    for file in files:
        shutil.copyfile(train_dir + el + '/' + file, data_dir + file)
        partition['train'].append(file[:file.find('.jpg')])
        labels[file[:file.find('.jpg')]] = el
    # test
    files = os.listdir(test_dir + el)
    for file in files:
        shutil.copyfile(test_dir + el + '/' + file, data_dir + file)
        partition['test'].append(file[:file.find('.jpg')])
        labels[file[:file.find('.jpg')]] = el
    # val
    files = os.listdir(val_dir + el)
    for file in files:
        shutil.copyfile(val_dir + el + '/' + file, data_dir + file)
        partition['val'].append(file[:file.find('.jpg')])
        labels[file[:file.find('.jpg')]] = el

# save partition, labels and label-info
with open('partition', 'wb') as pickle_out:
    pickle.dump(partition, pickle_out)

with open('labels', 'wb') as pickle_out:
    pickle.dump(labels, pickle_out)

with open('artists', 'wb') as pickle_out:
    pickle.dump(artists, pickle_out)
