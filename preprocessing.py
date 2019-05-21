import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from collections import Counter
import tensorflow as tf

labels_to_emotions = {0: 'Neutral', 1: 'Happy', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust',
                      6: 'Anger', 7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-Face'}
emotions_to_labels = {'Neutral': 0, 'Happy': 1, 'Sadness': 2, 'Surprise': 3, 'Fear': 4, 'Disgust': 5,
                      'Anger': 6, 'Contempt': 7}
folder_paths = ['Manually_Annotated_Images_', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8',
                'p9', 'p10', 'p11', 'p23']


def reconstruct_full_filepaths(base_path, base_filepaths, base_labels):
    filepaths = []
    labels = []
    for j, base_filepath in enumerate(base_filepaths):
        for i, folder_path in enumerate(folder_paths):
            if i == 0:
                continue
            if not os.path.exists(os.path.join(base_path, folder_paths[0] + folder_paths[i])):
                # print("Folder: ", os.path.join(base_path, folder_paths[0] + folder_paths[i]), " does not exist")
                continue
            filepath = os.path.join(base_path, folder_paths[0] + folder_paths[i], base_filepath)
            if os.path.isfile(filepath):
                basename, extension = os.path.splitext(os.path.basename(filepath))
                if extension.lower() == ".jpg" or extension.lower == ".png" or extension.lower() == ".jpeg":
                    filepaths.append(filepath)
                    labels.append(base_labels[j])
                    break
    return filepaths, labels


def construct_input(base_path="../Turquino/data/affectNet",
                    csv_path="../Turquino/data/affectNet/reduced_training_dlib.csv",
                    test_percentage=0.1, save=False, filepaths_save_path="data/filepaths.txt",
                    labels_save_path="data/labels.txt"):
    positive_useful_df = pd.read_csv(csv_path, index_col=1)
    positive_useful_df = positive_useful_df.reset_index()
    print("Finished reading csv ...")

    # remove 'contempt' and 'disgust' from the dataset
    positive_useful_df_2 = positive_useful_df[
        (positive_useful_df['expression'] != 5) & (positive_useful_df['expression'] != 7)]

    base_filepaths = positive_useful_df_2['subDirectory_filePath'].tolist()
    base_labels = positive_useful_df_2['expression'].tolist()

    filepaths, labels = reconstruct_full_filepaths(base_path, base_filepaths, base_labels)
    print("Finished reconstructing the full filepaths ...")
    print("Labels: ", list(set(labels)))

    print("\n Original filepaths and labels size: ", len(filepaths), len(labels))
    new_filepaths, new_labels = check_filepaths(filepaths, labels)
    print("\n After filepaths and labels size: ", len(new_filepaths), len(new_labels))
    print("Labels after: ", list(set(new_labels)))

    if save:
        with open(filepaths_save_path, "w") as filepaths_f:
            for filepath in new_filepaths:
                filepaths_f.write(filepath + "\n")
        with open(labels_save_path, "w") as labels_f:
            for label in new_labels:
                labels_f.write("%i\n" % label)
        print("\nFinished saving to files ...")
        return None
    else:
        return get_train_test_filenames_and_labels(filepaths, labels, test_percentage=test_percentage)


def load_filenames_and_labels(filenames_path="data/filepaths.txt", labels_path="data/labels.txt"):
    with open(filenames_path, "r") as filepaths_f:
        filenames = filepaths_f.readlines()
    filenames = [x.strip() for x in filenames]
    with open(labels_path, "r") as labels_f:
        labels = labels_f.readlines()
    labels = [int(x.strip()) for x in labels]
    print("Filenames: ", len(filenames))
    print("Labels: ", len(labels))
    return filenames, labels


def get_train_test_filenames_and_labels(filepaths, labels, test_percentage=0.1):
    np_filepaths = np.array(filepaths)
    np_labels = np.array(labels)
    print("Shape filepaths: ", np_filepaths.shape)
    print("Shape labels: ", np_labels.shape)

    train_filepaths, test_filepaths, train_labels, test_labels = train_test_split(np_filepaths, np_labels,
                                                                                  test_size=test_percentage,
                                                                                  random_state=42)

    print("\nDetails about the labels: ")
    print(" labels: ", Counter(np_labels))
    print(" labels train: ", Counter(train_labels))
    print(" labels test: ", Counter(test_labels))
    return train_filepaths, test_filepaths, train_labels, test_labels


def check_filepaths(filepaths, labels):
    new_filepaths = []
    new_labels = []
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print("Finished ", i)
        image_size = 224
        try:
            img_raw = tf.io.read_file(filepath)
            # img_tensor = tf.image.decode_image(img_raw) <-- doesn't provide image shape?
            img_tensor = tf.image.decode_jpeg(img_raw)
            img_resized = tf.image.resize(img_tensor, [image_size, image_size])
            img_normalized = (tf.cast(img_resized, tf.float32) / 127.5) - 1
            new_filepaths.append(filepath)
            new_labels.append(labels[i])
        except tf.errors.InvalidArgumentError as e:
            print(i, " - ", filepath, " error occured: {}".format(e))
    return new_filepaths, new_labels


if __name__ == '__main__':
    construct_input(save=True)
    # filenames,labels = load_filenames_and_labels()
    # get_train_test_filenames_and_labels(filenames, labels)
