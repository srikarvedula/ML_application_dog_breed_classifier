import os
import pandas as pd
import time
from tqdm import tqdm
from glob import glob
import numpy as np
from keras.preprocessing import image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from keras import layers
from keras import models
import math
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications import vgg16, vgg19, resnet50, mobilenet
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import keras.backend as K
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path
import tensorflow_addons as tfa



def path_to_tensor(file_path):
    img=image.load_img(file_path,target_size=(224,224))
    x=image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(file_paths):
    list_of_tensors=[path_to_tensor(file_path) for file_path in tqdm(file_paths)]

def load_paths(root_path='C:/Users/srika/PycharmProjects/dog_classifier/dataset/', split='train'):
    images = []
    labels = []
    label_to_class={}

    for dog_breed in os.listdir(os.path.join(root_path, split)):
        cur_images = glob(os.path.join(root_path, split, dog_breed, '*'))
        cur_labels = [int(dog_breed.split('.')[0])-1] * len(cur_images)
        label_to_class[int(dog_breed.split('.')[0])-1]=dog_breed.split('.')[1]
        images.extend(cur_images)
        labels.extend(cur_labels)
    assert len(images) == len(labels)
    print('Length of {} split: {}'.format(split, len(images)))
    return np.asarray(images), np.asarray(labels),label_to_class

def data_transform(image_file, img_size=256):
    training_list = np.stack([np.asarray(Image.open(l).convert('RGB').resize((img_size, img_size))) for l in image_file], axis=0)
    train_mean = np.mean(np.mean(np.mean(training_list, axis=0), axis=0), axis=0)
    train_images = (training_list - train_mean[None, None, None, :]) / 256.0
    return train_images

def CNN_model(input_shape, filters, n_hidden, nb_classes):
    resnet = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')
    print(resnet.layers)
    output = resnet.layers[-1].output
    print(output.shape)
    output = tf.keras.layers.Flatten()(output)
    print(output.shape)
    resnet = Model(resnet.input, output)

    model = models.Sequential()
    model.add(resnet)
    model.add(Dense(n_hidden, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.summary()




   #  model = models.Sequential()
   #  model.add(layers.Conv2D(filters[0], (3, 3), activation='relu', input_shape=input_shape))
   #  #model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
   #  #model.add(layers.BatchNormalization())
   #  model.add(layers.MaxPooling2D((2, 2)))
   #  model.add(layers.Conv2D(filters[1], (3, 3), activation='relu'))
   #
   #  #model.add(layers.BatchNormalization())
   #  model.add(layers.MaxPooling2D((2, 2)))
   #  model.add(layers.Conv2D(filters[2], (3, 3), activation='relu'))
   #
   #  #model.add(layers.BatchNormalization())
   #  # model.add(layers.MaxPooling2D((2, 2)))
   #  # model.add(layers.Conv2D(filters[3], (3, 3), activation='relu'))
   #
   #  #model.add(layers.BatchNormalization())
   #  #model.add(layers.MaxPooling2D((2, 2)))
   #
   #  # model.add(layers.Conv2D(filters[3], (3, 3), activation='relu'))
   #  # model.add(layers.MaxPooling2D((2, 2)))
   #
   # # model.add(layers.GlobalAvgPool2D())
   #
   #  model.add(Dense(n_hidden, activation='relu'))
   #  model.add(Dense(nb_classes, activation='softmax'))
    return model

def main():
    root_path = 'C:/Users/srika/PycharmProjects/dog_classifier/dataset'
    train_images, train_labels,label_to_class = load_paths(root_path, 'train')
    val_images, val_labels,_ = load_paths(root_path, 'valid')
    test_images, test_labels,_ = load_paths(root_path, 'test')
    NUM_CLASSES = 133
    train_labels= tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
    val_labels = tf.keras.utils.to_categorical(val_labels, NUM_CLASSES)
    test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)
    print("trai labels: ",train_labels)
    print("val labels: ", val_labels)
    norm_train_images=data_transform(train_images, 64)
    norm_val_images = data_transform(val_images, 64)
    norm_test_images = data_transform(test_images, 64)
    #train_mean = np.mean(np.mean(np.mean(train_images, axis=0), axis=0), axis=0)
    #print(train_mean)

    IMG_rows = 64
    IMG_cols = 64
    channel = 3

    N_HIDDEN = 1024
    FILTERS = [32,64,128,128]

    input_shape = (IMG_rows, IMG_cols, channel)
    cnn_model = CNN_model(input_shape, FILTERS, N_HIDDEN, NUM_CLASSES)
    cnn_model.summary()

    # opt = Adam(learning_rate=0.01)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                  restore_best_weights=False
                                                  )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.3,
                                                     patience=5,
                                                     verbose=1,
                                                     min_delta=1e-3, min_lr=1e-7,
                                                     )
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)

    cnn_model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy',tfa.metrics.F1Score(num_classes=NUM_CLASSES)])
    history = cnn_model.fit(norm_train_images, train_labels, batch_size=5, epochs=10,callbacks=[early_stop,reduce_lr],validation_data=(norm_val_images, val_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.4, 1])
    plt.title('Validation Curve')
    plt.legend(loc='lower right')
    plt.savefig('Validation_curve.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close("all")
    # Register the model
    cnn_model.save('dog_classifier_2.h5')

    output = cnn_model.predict(norm_test_images)
    preds = [np.argmax(l) for l in output]
    cls_preds = [label_to_class[l] for l in preds]
    print(test_images)
    print(cls_preds)

if __name__ == '__main__':
    main()