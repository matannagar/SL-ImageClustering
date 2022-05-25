import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import numpy as np
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Training Params.
trainWeights = 'trainWeights.h5'  # weights to save
epochs = 50
# Dataset folders
trainFolder = 'images/train/'
validFolder = 'images/validation/'
testFolder = 'images/test/'
# Image Dimension
imgDim = 28
# classes = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y'.split()
classes = 'B H W'.split()


# HyperParams
nbatch = 1  # 32 default. Number of samples to propagate each epoch.
learnRate = 0.001

class CustomCallback(keras.callbacks.Callback):
    """
    Custom callback class in order to save weights after each epoch.
    Was used to backup weights via Azure Virtual Machine.
    """

    def on_epoch_end(self, epoch, logs=None):
        try:
            copyfile(trainWeights, "Temp/epoch" + str(epoch) + "_weights.h5")
        except OSError:
            pass
        return


def train_generator():
    """
    Train the CNN12 model by Loading Training and Validation data.
    At the end of the training a learning graph will be plotted.
    """

    # Load training data with augmentation.
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       # rotation_range=10,  # randomly rotate up to 40 degrees.
                                       # width_shift_range=0.2,  # randomly shift range.
                                       # height_shift_range=0.2,
                                       # shear_range=0.2,
                                       # zoom_range=0.2,
                                       # fill_mode="nearest"
                                       )  # fill new pixels created by shift

    train_generator = train_datagen.flow_from_directory(trainFolder,
                                                        target_size=(imgDim, imgDim),
                                                        color_mode='grayscale',
                                                        batch_size=nbatch,
                                                        classes=classes,
                                                        class_mode="categorical")

    # with open('x_train_generator.json', 'w') as f:
    #     json.dump(next(train_generator)[0].tolist(), f)

    return train_generator

def valid_generator():

    # # Load validation data (10% of original train data).

    valid_datagen = ImageDataGenerator(rescale=1. / 255.)

    valid_generator = valid_datagen.flow_from_directory(testFolder,
                                                        target_size=(imgDim, imgDim),
                                                        color_mode='grayscale',
                                                        batch_size=nbatch,
                                                        classes=classes,
                                                        class_mode="categorical")

    return valid_generator

if __name__ == '__main__':
    train_generator = train_generator()
    # print(train_generator)
    print(train_generator[0].shape)
    print(train_generator[1].shape)
    # print(train_generator[0][0].shape)
    # print(train_generator[0][0][0].shape)

    # x_train = []
    # y_train = []
    # for x, y in train_generator:
    #     x_train.append(x)
    #     y_train.append(y)
    #
    # print(x_train[0].shape)
    # print(y_train[0].shape)