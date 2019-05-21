import tensorflow as tf
import keras
import random
import numpy as np
import cv2
import argparse
import datetime
import pandas as pd
import os
from sklearn.utils import class_weight
from keras import backend as K

from preprocessing import load_filenames_and_labels, get_train_test_filenames_and_labels


class DataGenerator():
    def __init__(self, filenames, labels, batch_size=32, num_classes=6, input_shape=(224, 224),
        with_categorical=True, with_mean_sub=True, shuffle=True):
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.with_categorical = with_categorical
        self.with_mean_sub = with_mean_sub
        self.shuffle = shuffle

    def process_img(self, img, with_mean_sub=True):
        new_img = img.astype(np.float32)
        new_img /= 127.5
        new_img -= 1.
        if with_mean_sub:
            # Mean is an array of three elements obtained by the average of R, G, B pixels of all images obtained from ImageNet. 
            # The values for Imagenet are : [ 103.939, 116.779, 123.68 ]
            mean = [103.939, 116.779, 123.68]
            new_img[:, :, 0] -= mean[0]
            new_img[:, :, 1] -= mean[1]
            new_img[:, :, 2] -= mean[2]
        return new_img

    def generator(self):
        x_train = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3))
        # y_train = np.zeros((self.batch_size, self.num_classes))
        while True:
            number_of_batches = int(np.floor(len(self.filenames) / self.batch_size))
            # 1. Shuffle filenames in case the filenames don't fall into exact batches
            permutation_indices = list(range(len(self.filenames)))
            if self.shuffle:
                random.shuffle(permutation_indices)
            local_filenames = [self.filenames[i] for i in permutation_indices]
            local_labels = [self.labels[i] for i in permutation_indices]
            for i in range(number_of_batches):
                for j in range(int(i*self.batch_size), int((i+1)*self.batch_size)):
                    img = cv2.imread(local_filenames[j])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img.shape[0:2] != self.input_shape:
                        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_AREA)
                    img = self.process_img(img, with_mean_sub=self.with_mean_sub)
                    x_train[j - i*self.batch_size] = img
                if self.with_categorical:
                    y_train = keras.utils.to_categorical(local_labels[
                        int(i*self.batch_size): int((i+1)*self.batch_size)], self.num_classes)
                else:
                    y_train = local_labels[int(i*self.batch_size): int((i+1)*self.batch_size)]
                y_train = np.array(y_train)
                yield x_train, y_train


def test_DataGenerator():
    filenames, labels = load_filenames_and_labels()
    classes = list(set(labels))
    print("Classes: ", classes, labels[:20])
    labels = [x if x != 6 else 5 for x in labels]
    classes = list(set(labels))
    print("New classes: ", classes, labels[:20])
    train_filepaths, test_filepaths, train_labels, test_labels = get_train_test_filenames_and_labels(filenames, labels)
    
    print(train_filepaths)
    train_generator = DataGenerator(train_filepaths, train_labels, batch_size=4, num_classes=len(classes), 
        with_categorical=True, with_mean_sub=False)
    generator = train_generator.generator()

    dummy_x_train, dummy_y_train = next(generator)
    print("Dummy x train shape: ", dummy_x_train.shape)
    print("Dummy x train slice: ", dummy_x_train[0, :5, :5, 0])
    print("Dummy y train shape: ", dummy_y_train.shape)
    print("Dummy y train: ", dummy_y_train)


def main():
	# 0. Parse command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the training process")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    args = parser.parse_args() 

    # 1. Preparing data
    filenames, labels = load_filenames_and_labels()
    assert(len(filenames) == len(labels))
    labels = [x if x != 6 else 5 for x in labels]  # replacing label 6 with 5 to avoid confusion in the algorithms
    classes = list(set(labels))
    print("Classes: ", classes)
    train_filepaths, test_filepaths, train_labels, test_labels = get_train_test_filenames_and_labels(filenames, labels)
    assert(len(train_filepaths) == len(train_labels))
    assert(len(test_filepaths) == len(test_labels))

    image_shape = (224, 224, 3)
    train_datagen = DataGenerator(train_filepaths, train_labels, batch_size=args.batch_size, num_classes=len(classes),
        input_shape=(image_shape[0], image_shape[1]), with_categorical=True, with_mean_sub=True)
    test_datagen = DataGenerator(test_filepaths, test_labels, batch_size=args.batch_size, num_classes=len(classes),
        input_shape=(image_shape[0], image_shape[1]), with_categorical=True, with_mean_sub=True, shuffle=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    K.set_session(sess)

    # 2. Constructing model
    base_model = keras.applications.mobilenet.MobileNet(input_shape=image_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    # 2.1 version
    model = keras.Sequential([
        base_model,
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(len(classes), activation='softmax')
    ])

    # 2.2 version
    # model = keras.Sequential([
    #     base_model,
    #     keras.layers.MaxPooling2D((2, 2)),
    #     keras.layers.MaxPooling2D((2, 2)),
    #     keras.layers.Dense(32, activation='relu'),
    #     keras.layers.Dense(len(classes), activation='softmax')
    # ])

    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    check_model_summary = True
    if check_model_summary:
        # print(base_model.summary())
        print(model.summary())

    # 3. Initialize callbacks
    time_tag = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_dir = os.path.join("logs", time_tag)
    os.makedirs(log_dir)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    checkpoint_dir = os.path.join("checkpoints", time_tag)
    os.makedirs(checkpoint_dir)
    # 'checkpoints/weights.{epoch:02d}-loss_{val_loss:.2f}-acc_{val_acc:0.5f}.hdf5'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, 'weights.{epoch:02d}-acc_{val_acc:.5f}.hdf5'), monitor='val_acc', period=1)
    early_stopping_checkpoint = keras.callbacks.EarlyStopping(patience=5)
    reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, cooldown=1)

    # 4. Train
    steps_per_epoch = int(np.floor(len(train_filepaths) / args.batch_size))
    validation_steps = int(np.floor(len(test_filepaths) / args.batch_size))
    print("Steps per epoch: ", steps_per_epoch)
    print("Validations steps: ", validation_steps)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    class_weights_dict = {}
    for i in range(len(class_weights)):
        class_weights_dict[i] = class_weights[i]
    print("class weights dict: ", class_weights_dict)

    model.fit_generator(generator=train_datagen.generator(), steps_per_epoch=steps_per_epoch, epochs=args.epochs,
        validation_data = test_datagen.generator(), validation_steps=validation_steps, verbose=1, 
        class_weight=class_weights_dict,
        callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping_checkpoint, reduce_lr_on_plateau])

    history_dir = os.path.join('history', time_tag)
    os.makedirs(history_dir)
    pd.DataFrame(history).to_csv(os.path.join(history_dir, "history.csv"))


if __name__ == "__main__":
    # test_DataGenerator()
    main()
