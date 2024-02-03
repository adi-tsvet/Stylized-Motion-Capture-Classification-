from keras.optimizers import Adam

import conf
from networks.utilities import Utilities
import networks.custom_losses as custom_losses
from keras import callbacks
from keras.models import Sequential
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Lambda, Dropout
import logging
import os

logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.basename(__file__) + '.log',
                    format="{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
                    style="{")
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)


# embedding network and training
class SimilarityNetwork(Utilities):
    """
    The SimilarityNetwork class defines our own triplet similarity network.

    This class inherits from the `Utilities` class and is used to build, compile, and train a similarity learning model.

    Args:
        train_loader: The data loader for the training dataset.
        validation_loader: The data loader for the validation dataset.
        test_loader: The data loader for the test dataset.
        checkpoint_dir: The directory where model checkpoints will be saved.

    Attributes:
        train_set: The training dataset loader.
        validation_set: The validation dataset loader.
        test_set: The test dataset loader.
        exemplar_dim: The dimensions of the exemplar data.
        checkpoint_dir: The directory for saving model checkpoints.
        embedding_size: The size of the embedding produced by the network.
        callbacks: List of Keras callbacks for training.
    """

    def __init__(self, train_loader, validation_loader, test_loader, checkpoint_dir):
        super().__init__()
        # train_generator's job is to randomly, and lazily, load batches from disk
        self.train_set = train_loader
        self.validation_set = validation_loader
        self.test_set = test_loader
        self.exemplar_dim = train_loader.exemplar_dim
        self.checkpoint_dir = checkpoint_dir
        self.embedding_size = conf.embedding_size
        self.callbacks = [callbacks.EarlyStopping(monitor='val_batch_triplet_loss', patience=5, mode='min', verbose=1)]
        self._network = Sequential()
        self.build_model()
        self.compile_model()

    def build_model(self):
        """
        Build the architecture of the neural network.

        Add layers to the neural network model to create the desired architecture.

        Args:
            None

        Returns:
            None
        """
        # Add a new dimension for 2D convolution
        input_shape = (self.exemplar_dim[0], self.exemplar_dim[1], 1)
        self._network.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu',
                                 input_shape=input_shape))
        self._network.add(Dropout(0.2))
        self._network.add(BatchNormalization())
        self._network.add(MaxPool2D(2, 2))
        self._network.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
        self._network.add(Dropout(0.2))
        self._network.add(BatchNormalization())
        self._network.add(MaxPool2D(2, 2))
        self._network.add(Flatten())
        self._network.add(Dense(self.embedding_size))
        self._network.add(Dropout(0.2))
        # self._network.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1, epsilon=1e-13)))

    def compile_model(self):
        """
        Compile the neural network with a custom triplet loss function.

        Compiles the neural network using the Adam optimizer and a custom triplet loss function.
        Also sets up the network for training.

        Args:
            None

        Returns:
            None
        """
        opt = Adam(learning_rate=0.0001, beta_1=0.5)
        try:
            self._network.compile(optimizer=opt, loss=custom_losses.batch_triplet_loss,
                                  metrics=[custom_losses.batch_triplet_loss], run_eagerly=False)
        except RuntimeError:
            self._network.compile(optimizer=opt, loss=custom_losses.batch_triplet_loss, metrics=[
                custom_losses.batch_triplet_loss])
        finally:
            self._network.summary()

    def run_model_training(self, callbacks=None):
        """
       Train the neural network on the training dataset.

       Trains the neural network on the training dataset and uses the validation dataset for monitoring.

       Args:
           None

       Returns:
           None
       """
        if callbacks is None:
            self.callbacks = []
        self._network.fit(self.train_set, validation_data=self.validation_set, epochs=conf.n_similarity_epochs,
                          steps_per_epoch=self.train_set.__len__(),
                          validation_steps=self.validation_set.__len__(), callbacks=self.callbacks)

    def evaluate(self):
        """
        Evaluate the model on the test dataset.

        Evaluates the trained model on the test dataset attribute.

        Args:
            None

        Returns:
            None
        """
        self._network.evaluate(self.test_set, steps=self.test_set.__len__())
