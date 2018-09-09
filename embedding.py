#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf
import keras

from keras import backend as K
from keras.layers import Embedding, Input, dot, Dense, Flatten, Reshape

import input

EPOCHS = 2
BATCH_SIZE = 500
EMBEDDING_OUTPUT_SIZE = 300
SKIP_WINDOW=3
NUM_VALIDATION_TOKENS=16
NUM_SIMILAR_WORDS=8

tokenized_data, vocab = input.tokenized()
vocab = np.array(vocab)

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
    similarity = Reshape(())(similarity)
    dotprod = dot([x,y], axes=2, normalize=False)
    dotprod = Reshape((1,))(dotprod)
    output = Dense(1,activation="sigmoid")(dotprod)

    train_model = keras.Model(inputs=[target_word, context_word], outputs=output)
    validation_model = keras.Model(inputs=[target_word, context_word], outputs=similarity)

    return train_model, validation_model

train_model, validation_model = negative_sampling_model()

validation_tokens = list(range(24,4*16+1,4))#random.sample(range(1,len(vocab)), 16)

def get_token_similarities(tokens):
    word_target = np.array(tokens)
    word_context = np.empty_like(word_target)
    similarities = np.empty((len(tokens),len(vocab)))
    for i in range(1, len(vocab)):
        word_context.fill(i)
        similarities[:,i] = validation_model.predict_on_batch([word_target, word_context])
    return similarities

def find_similar_words(epoch, _):
    #if (epoch+1) % 4 != 0: return

    similarities = get_token_similarities(validation_tokens)
    sorted_tokens = similarities.argsort(axis=1)
    top_k = sorted_tokens[:,-1:-2-NUM_SIMILAR_WORDS:-1]
    print(vocab[top_k])

train_model.compile("rmsprop", "binary_crossentropy")

train_model.fit(x=word_pairs,
                y=labels,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=find_similar_words)])
