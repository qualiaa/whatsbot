#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import keras

from keras import backend as K
from keras.layers import Embedding, Input, dot, Dense, Flatten, Reshape

import input

EPOCHS = 20
BATCH_SIZE = 500
EMBEDDING_OUTPUT_SIZE = 300
SKIP_WINDOW=3

tokenized_data, vocab = input.tokenized()

dataset = tf.data.Dataset.from_tensor_slices(tokenized_data)

sampling_table = keras.preprocessing.sequence.make_sampling_table(len(vocab))
word_pairs, labels = keras.preprocessing.sequence.skipgrams(tokenized_data,
        len(vocab),
        sampling_table=sampling_table,
        window_size=SKIP_WINDOW,
        shuffle=True)

word_pairs = list(np.expand_dims(np.array(a, dtype=np.int16), 1) for a in zip(*word_pairs))
labels = np.expand_dims(np.array(labels, dtype=np.int16), 1)

def negative_sampling_model():
    target_word = Input((1,))
    context_word = Input((1,))
    embedding = Embedding(len(vocab), EMBEDDING_OUTPUT_SIZE,
            input_length=1,
            mask_zero=True)

    x = embedding(target_word)
    y = embedding(context_word)
    similarity = dot([x,y], axes=2, normalize=True)
    dotprod = dot([x,y], axes=2, normalize=False)
    dotprod = Reshape((1,))(dotprod)
    output = Dense(1,activation="sigmoid")(dotprod)

    train_model = keras.Model(inputs=[target_word, context_word], outputs=output)

    return train_model

train_model = negative_sampling_model()

train_model.compile("rmsprop", "binary_crossentropy")

train_model.fit(x=word_pairs, y=labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
