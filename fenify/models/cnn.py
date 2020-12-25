from fenify.helpers.visualize import visualize_samples
from .base import Model
from fenify.prepare.sample import PieceSample
import tensorflow as tf
import numpy as np
from fenify.helpers.utils import fen_to_piece_list
from fenify.helpers.visualize import visualize_samples
from fenify.helpers.image import resize_image
import uuid
import os
from shutil import copyfile, copytree
import json
import tempfile

class CNNModel(Model):
    DROPOUT = .03
    def __init__(self):
        self.model = tf.keras.models.Sequential([
                        tf.keras.layers.Conv2D(input_shape=(28, 28, 3), filters=32, kernel_size=3, activation="relu"),
                        tf.keras.layers.Dropout(self.DROPOUT),
                        tf.keras.layers.Conv2D(input_shape=(28, 28, 3), filters=32, kernel_size=3, activation="relu"),
                        tf.keras.layers.Dropout(self.DROPOUT),
                        tf.keras.layers.Conv2D(input_shape=(28, 28, 3), filters=32, kernel_size=3, activation="relu"),
                        tf.keras.layers.Dropout(self.DROPOUT),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dropout(self.DROPOUT),
                        tf.keras.layers.Dense(128, activation="relu"),
                        tf.keras.layers.Dropout(self.DROPOUT),
                        tf.keras.layers.Dense(13),
                    ])
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        self.name = "cnn"
        self.models_dir = os.path.join("models_data", self.name)
        self._init_model_dir()
        self._init_tensorboard()
        self.x_train, self.y_train = [], []
        self.x_test, self.y_test = [], []
        self._acc = None
        self._confusion = None

    def _init_model_dir(self):
        os.makedirs(self.models_dir, exist_ok=True)
        all_models = os.listdir(self.models_dir)
        latest_model = 0
        for m in all_models:
            if int(m) > latest_model:
                latest_model = int(m)
        latest_model += 1
        self.model_data_path = os.path.join(self.models_dir, str(latest_model))
        os.makedirs(self.model_data_path)
        self.model_path = os.path.join(self.model_data_path, "model.hdf5")

    def _init_tensorboard(self):
        self.log_dir = os.path.join(self.model_data_path, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

    def _convert_samples_to_set(self, pieces_samples):
        pieces = 'rbnkqpRBNKQP'
        x_train, y_train = [], []
        for p in pieces_samples:
            x_train.append(p.image)
            if p.piece_type == "":
                y_train.append(12)
            else:
                y_train.append(pieces.find(p.piece_type))
        return np.array(x_train), np.array(y_train)

    def add_samples(self, pieces_samples):
        self.x_train, self.y_train = self._convert_samples_to_set(pieces_samples)
        
    def train(self, epochs):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                        save_weights_only=False,
                                                        save_best_only=True,
                                                        monitor='loss',
                                                        verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, callbacks=[cp_callback, tensorboard_callback])
    
    def evaluate(self, pieces_samples):
        n = len(pieces_samples)
        self.x_test, self.y_test = self._convert_samples_to_set(pieces_samples)
        probability_model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])
        y_pred = probability_model(self.x_test)
        labels = np.argmax(y_pred, axis=-1)
        labels_resize = np.reshape(labels, (n, 1))
        ys_resize = [[x] for x in self.y_test]
        m = tf.keras.metrics.Accuracy()
        m.update_state(ys_resize, labels_resize)
        self._acc = float(m.result().numpy())
        self._confusion = self._calc_confusion(self.y_test, labels)
        self._miscolored = self._calc_miscolored(self.y_test, labels)

    def _calc_confusion(self, truth, pred):
        mat = [[0 for _ in range(13)] for _ in range(13)]
        for i in range(len(truth)):
            mat[pred[i]][truth[i]] += 1
        return mat

    def _calc_miscolored(self, truth, pred):
        cnt = 0
        tot = len(truth)
        for i in range(len(truth)):
            if truth[i] != 13 and abs(truth[i] - pred[i]) == 6:
                cnt += 1
        return cnt / tot
    

    def save_snapshot(self):
        metrics = {
            "accuracy": self.accuracy(),
            "miscolored": self.miscolored(),
            "confusiont matrix": self.confusion_matrix()
        }
        with open(os.path.join(self.model_data_path, 'metrics.json'), 'w+') as f:
            f.write(json.dumps(metrics, indent=2))

        with open(os.path.join(self.model_data_path, 'metadata.json'), 'w+') as f:
            f.write(json.dumps(self.metadata(), indent=2))


    def load_latest(self):
        models_dir = os.path.join("models_data", self.name)
        os.makedirs(models_dir, exist_ok=True)
        all_models = os.listdir(models_dir)
        latest_model = 0
        for m in all_models:
            if int(m) > latest_model and os.path.isfile(os.path.join(models_dir, str(m), 'model.hdf5')):
                latest_model = int(m)
        model_data_path = os.path.join(models_dir, str(latest_model))
        if latest_model != 0:
            print(f"LATEST: {latest_model}")
            self.model = tf.keras.models.load_model(os.path.join(model_data_path, "model.hdf5"))
            self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    def metadata(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)

        return {
            "name": self.name,
            "description": "The same model as the one used in mnist in tf quick start guide",
            "summary": short_model_summary,
        }

    def confusion_matrix(self):
        return self._confusion

    def accuracy(self):
        return self._acc
        
    def miscolored(self):
        return self._miscolored

    def get_samples(self, board_sample):
        piece_samples = []
        pieces_list = fen_to_piece_list(board_sample.fen)
        l = board_sample.image.shape[0] // 8
        for i in range(8):
            for j in range(8):
                idx = i * 8 + j
                color = (i + j) % 2
                segmented_image = board_sample.image[l * i: l * (i + 1), l * j: l * (j + 1)]
                resized_segmented_image = resize_image(segmented_image, (28, 28))
                piece_sample = PieceSample(resized_segmented_image, pieces_list[idx], board_sample.pieces_type, board_sample.board_type, color, board_sample.pieces_distortions[idx])
                piece_samples.append(piece_sample)
        return piece_samples