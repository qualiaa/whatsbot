#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, Softmax, Activation
from tensorflow.keras.layers import TimeDistributed, LSTM


import input
import embedding

BATCH_SIZE=20
WORDS_PER_SEQUENCE=30

def _test_train_split(corpus, vocab, train_ratio):
    """ rather than split exactly at train_ratio, find the next End-of-Message
    (EOM) token """
    test_start_index = int(train_ratio*len(corpus))
    train_corpus  = corpus[:test_start_index]
    test_corpus = corpus[test_start_index:]
    
    eom_token = vocab.index("EOM")
    next_eom_index = test_corpus.index(eom_token)
    train_corpus.extend(test_corpus[:next_eom_index+1])
    test_corpus = test_corpus[next_eom_index+1:]

    print(len(train_corpus),len(test_corpus))
    return train_corpus, test_corpus

def model(words_per_sequence, vocab_size, embedding_weights_path="weights/embedding.npy"):
    embedding_weights = embedding.load(embedding_weights_path).squeeze()
    print(embedding_weights.shape)
    embedding_layer = Embedding(
            embedding_weights.shape[0],
            embedding_weights.shape[1],
            input_length=words_per_sequence,
            weights=[embedding_weights])
    print(np.array(embedding_layer.get_weights()).squeeze().shape)
    """ Calling embedding_layer.set_weights has been decreed a sin.
    You have to use an undocumented weights argument. The weights should be
    wrapped in a list as anything else would be against God """
    #embedding_layer.set_weights(embedding_weights)

    layer_input = Input((words_per_sequence,))
    x = embedding_layer(layer_input)
    print(x.shape)

    x = LSTM(embedding_weights.shape[-1], return_sequences=True)(x)
    print(x.shape)
    x = LSTM(embedding_weights.shape[-1], return_sequences=True)(x)
    print(x.shape)
    x = Dropout(0.5)(x)
    print(x.shape)
    x = TimeDistributed(Dense(vocab_size))(x)
    print(x.shape)
    x = Activation("softmax", name="softmax")(x)
    print(x.shape)

    return keras.Model(inputs=[layer_input], outputs=[x])

def to_categorical(num_categories):
    def inner(x, y):
        return x, tf.one_hot(y,num_categories)
    return inner

def train():
    tokenized_corpus, vocab = input.tokenized()

    train_batches, test_batches = [tf.data.Dataset.from_tensor_slices(arr)
            .apply(input.to_rnn_input(BATCH_SIZE, WORDS_PER_SEQUENCE, len(arr)))
            .map(to_categorical(len(vocab)))
        for arr in _test_train_split(tokenized_corpus, vocab, 0.8)]


    train_model = model(WORDS_PER_SEQUENCE, len(vocab))
    for l in train_model.layers:
        print (l.input_shape, l.output_shape)
    train_model.compile("rmsprop","categorical_crossentropy")

    print(train_batches.output_shapes)
    train_model.fit(train_batches, steps_per_epoch=100)
    

if __name__ == "__main__":
    train()
