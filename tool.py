from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.callbacks import Callback
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import itertools 
import cv2
import os
import time


def load_data(data_path, labels, image_size):

    x_train = [] # training images.
    y_train  = [] # training labels.
    x_test = [] # testing images.
    y_test = [] # testing labels.


    for label in labels:
        trainPath = os.path.join(data_path, 'train',label)
        for file in tqdm(os.listdir(trainPath)):
            image = cv2.imread(os.path.join(trainPath, file))
            image = cv2.bilateralFilter(image, 2, 50, 50) # remove images noise.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_size, image_size))
            x_train.append(image)
            y_train.append(labels.index(label))
        
        testPath = os.path.join(data_path, 'val',label)
        for file in tqdm(os.listdir(testPath)):
            image = cv2.imread(os.path.join(testPath, file))
            image = cv2.bilateralFilter(image, 2, 50, 50)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_size, image_size))
            x_test.append(image)
            y_test.append(labels.index(label))


    x_train = np.array(x_train) / 255.0 # normalize Images into range 0 to 1.
    x_test = np.array(x_test) / 255.0

    x_train, y_train = shuffle(x_train, y_train, random_state=101)

    images = [x_train[i] for i in range(15)]
    fig, axes = plt.subplots(3, 5, figsize = (10, 10))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

     
    y_train = tf.keras.utils.to_categorical(y_train) #One Hot Encoding on the labels
    y_test = tf.keras.utils.to_categorical(y_test)

    print(x_test.shape)
    print(x_train.shape)
    print(y_test.shape)
    print(y_train.shape)


    return x_train, x_test, y_train, y_test


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



def draw_curve(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(history.epoch) + 1)

    plt.figure(figsize=(15,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Set')
    plt.plot(epochs_range, val_acc, label='Val Set')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Set')
    plt.plot(epochs_range, val_loss, label='Val Set')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')

    plt.tight_layout()
    plt.show()


def draw_matrix(label, y_test_new, pred, acc):
    
    print("Test Accuracy: ",np.round(acc*100,2))
    print(classification_report(y_test_new,pred,target_names = label,digits = 4))
    confusion_mtx = confusion_matrix(y_test_new,pred)
    cm = plot_confusion_matrix(confusion_mtx, target_names = label, normalize=False)


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.train_start_time = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - self.epoch_start_time
        self.times.append(elapsed_time)
        print("Epoch {}: {:.2f} seconds".format(epoch + 1, elapsed_time))

    def on_train_end(self, logs={}):
        train_end_time = time.time()
        total_time = train_end_time - self.train_start_time
        print("Total training time: {:.2f} seconds".format(total_time))