import keras
import tensorflow as tf
import concurrent
import random
import os.path
import numpy as np
from glob import glob
import conf


class SimilarityDataLoader(keras.utils.Sequence):
    def __init__(self, dict_similarity_exemplars, shuffle=True):
        print("initializing similarity data loader")
        self.exemplar_dim = conf.similarity_exemplar_dim
        self.batch_size = conf.similarity_batch_size
        self.shuffle = shuffle
        self.dict_similarity_exemplars = dict_similarity_exemplars
        self.class_indexes, self.list_class_tuples = zip(
            *[(index + 1, value) for index, value in enumerate(list(self.dict_similarity_exemplars.keys()))])
        self.num_classes = len(self.class_indexes)
        self.class_indexes = list(self.class_indexes)
        self.list_class_tuples = list(self.list_class_tuples)
        # self.num_batches = len(self.dict_similarity_exemplars[self.list_class_tuples[0]])
        self.num_batches = len(self.dict_similarity_exemplars[next(iter(self.dict_similarity_exemplars.keys()))])
        print(f"num batches: {self.num_batches}")
        self.exemplar_idx = random.randint(0, self.num_batches-1)

    def unison_shuffling(self):
        # generate random batch num index, to be applied to all dict class lists, and unison shuffle
        # class and class_idx order (a within batch shuffling)
        self.exemplar_idx = random.randint(1, self.num_batches-1)
        p = list(np.random.permutation(self.num_classes))
        self.list_class_tuples = [self.list_class_tuples[i] for i in p]
        self.class_indexes = [self.class_indexes[i] for i in p]

    def on_epoch_end(self):
        if self.shuffle is True:
            self.unison_shuffling()

    def __len__(self):
        # num_batches
        return self.num_batches

    def __getitem__(self, index):
        # batch features shape: tensor: [ 56 100  91],  batch labels shape: tensor: [56]
        batch_features = tf.stack([self.dict_similarity_exemplars[class_tuple][self.exemplar_idx] for class_tuple in
                                   self.list_class_tuples])
        batch_labels = tf.constant(self.class_indexes)
        tf.debugging.assert_shapes([(batch_features, (conf.similarity_batch_size, conf.similarity_exemplar_dim[0],
                                                      conf.similarity_exemplar_dim[1]))],
                                   message="Similarity batch feature erroneous shape}")
        return batch_features, batch_labels
