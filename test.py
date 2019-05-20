import tensorflow as tf
import numpy as np
import cv2

from preprocessing import labels_to_emotions


def run_on_camera(img_size=480, save_pb=False):
    cap = cv2.VideoCapture(0)
    print("Default frame width and height: ", cap.get(3), cap.get(4))
    cap.set(3, 640)
    cap.set(4, 480)
    print("New frame width and height: ", cap.get(3), cap.get(4))

    model = tf.keras.models.load_model('checkpoints/checkpoints/2019_05_19-21_33_06/weights.08-acc_0.67565.hdf5')
    print("Model summary: ", model.summary())

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        if 224 < frame.shape[1]:
            # for shrinking use cv2.INTER_AREA
            copy_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        else:
            # for upscaling use cv2.INTER_LINEAR or cv2.INTER_CUBIC (slower)
            copy_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
        copy_frame = cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGB)
        copy_frame = copy_frame.astype(np.float32)
        copy_frame = (copy_frame / 127.5) - 1
        copy_frame = np.expand_dims(copy_frame, axis=0)
        # print("Frame shape: ", copy_frame.shape, copy_frame.dtype)
        tf_frame = tf.convert_to_tensor(copy_frame, dtype=tf.float32)
        # print("TF frame shape: ", tf_frame.shape)
        prediction = model(tf_frame)

        if save_pb:
            tf.saved_model.save(model, "tmp/mymodel/1/")
            save_pb=False

        print("Prediction: ", type(prediction), prediction)
        np_prediction = np.array(prediction)
        emotion = labels_to_emotions[np.argmax(np_prediction)]
        print("Emotion: ", emotion)

        cv2.imshow('Emotio', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_on_image():
    pass


def save_attempt():
    model = tf.keras.models.load_model('checkpoints/checkpoints/2019_05_19-21_33_06/weights.08-acc_0.67565.hdf5')
    print("Model summary: ", model.summary())
    tf.saved_model.save(model, "/tmp/mymodel/1/")


if __name__ == "__main__":
    run_on_camera()
    # save_attempt()
