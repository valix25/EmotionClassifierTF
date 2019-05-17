import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import datetime
import os
import numpy as np
import pandas as pd
# import keras_applications

from preprocessing import load_filenames_and_labels, get_train_test_filenames_and_labels


def process_input_sample(filename, label):
    image_size = 224
    img_raw = tf.io.read_file(filename)
    # img_tensor = tf.image.decode_image(img_raw) <-- doesn't provide image shape?
    img_tensor = tf.image.decode_jpeg(img_raw)
    img_resized = tf.image.resize(img_tensor, [image_size, image_size])
    img_normalized = (tf.cast(img_resized, tf.float32) / 127.5) - 1
    return img_normalized, label


def main():
    # 0. Parse command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the training process")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    args = parser.parse_args()

    # 1. Load the filepaths and the labels
    filenames, labels = load_filenames_and_labels()
    train_filepaths, test_filepaths, train_labels, test_labels = get_train_test_filenames_and_labels(filenames, labels)
    classes = list(set(labels))

    # Test process_input_sample
    check_processed = False
    if check_processed:
        img_processed_test, label_processed_test = process_input_sample(train_filepaths[0], train_labels[0])
        print("Image processed test shape: ", img_processed_test.shape)
        print("Image processed test type: ", img_processed_test.dtype)
        print("Label processed test: ", label_processed_test)
        plt.figure()
        plt.imshow(img_processed_test)
        plt.show()

    # 2. Construct training and test sets (NOT a balanced way of feeding the data)
    # check tf.data.Dataset.from_generator to use with BalancedBatchGenerator?
    train_data = tf.data.Dataset.from_tensor_slices((train_filepaths, train_labels))
    train_data = train_data.shuffle(buffer_size=len(train_filepaths))
    train_data = train_data.map(process_input_sample)
    train_data = train_data.repeat().batch(args.batch_size)
    # 'prefetch' lets the dataset fetch batches, in the background while the model is training.
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_data = tf.data.Dataset.from_tensor_slices((test_filepaths, test_labels))
    test_data = test_data.map(process_input_sample).batch(args.batch_size)
    test_data = test_data.repeat()

    check_train_batch = False
    if check_train_batch:
        for example_batch, labels_batch in train_data:
            print("Example batch shape: ", example_batch.shape)
            print("Labels batch shape: ", labels_batch.shape)
            break

    # 3. Set up model
    image_shape = (224, 224, 3)
    base_model = tf.keras.applications.MobileNet(input_shape=image_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    check_model_summary = False
    if check_model_summary:
        # print(base_model.summary())
        print(model.summary())

    # 4. Initialize callbacks
    time_tag = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_dir = os.path.join("logs", time_tag)
    os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_dir = os.path.join("checkpoints", time_tag)
    os.makedirs(checkpoint_dir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/weights.{epoch:02d}-loss_{val_loss:.2f}-acc_{val_acc:0.5f}.hdf5', period=1)
    early_stopping_checkpoint = tf.keras.callbacks.EarlyStopping(patience=5)

    # 5. Training the model
    steps_per_epoch = round(train_filepaths.shape[0]) // args.batch_size
    validations_steps = round(test_filepaths.shape[0]) // args.batch_size
    print("Steps per epoch: ", steps_per_epoch)
    print("Validations steps: ", validations_steps)
    # there is class weight to for balancing
    history = model.fit(train_data, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                        validation_data=test_data, validations_steps=validations_steps,
                        callbacks=[model_checkpoint_callback, tensorboard_callback, early_stopping_checkpoint])

    history_dir = os.path.join('history', time_tag)
    os.makedirs(history_dir)
    pd.DataFrame(history).to_csv(os.path.join(history_dir, "history.csv"))


if __name__ == "__main__":
    # help(keras_applications.mobilenet_v2.preprocess_input)
    main()
