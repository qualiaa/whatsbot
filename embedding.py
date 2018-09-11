#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf
import keras as keras

from keras.layers import Embedding, Input, dot, Dense, Flatten, Reshape, Lambda

import input

TOTAL_TRAIN_EXAMPLES=int(1e6)
EXAMPLES_PER_EPOCH=int(5e5)
BATCH_SIZE = 500
EMBEDDING_OUTPUT_SIZE = 300
SKIP_WINDOW=3
NUM_VALIDATION_TOKENS=16
NUM_SIMILAR_WORDS=8

STEPS_PER_EPOCH=EXAMPLES_PER_EPOCH // BATCH_SIZE
EPOCHS = TOTAL_TRAIN_EXAMPLES // EXAMPLES_PER_EPOCH

def negative_sampling_model(input_size, output_size):
    """ necessary for tf.data.Dataset because nested inputs are broken
    joined_input = Input((2,1))
    chan0 = Lambda(lambda x: x[:,0,:], output_shape=(1,))
    chan1 = Lambda(lambda x: x[:,1,:], output_shape=(1,))
    """
    target_word = Input((1,))
    context_word = Input((1,))
    embedding = Embedding(input_size, output_size,
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

class SimilarityCallback(keras.callbacks.Callback):
    def __init__(self, vocab, validation_tokens, model):
        super().__init__()
        self.vocab = vocab
        self.validation_model = model
        self.validation_tokens = validation_tokens

    def on_epoch_end(self, epoch, _):
        self._find_similar_words(epoch)
        
    def _get_token_similarities(self, tokens):
        word_target = np.array(tokens)
        word_context = np.empty_like(word_target)
        similarities = np.empty((len(tokens),len(self.vocab)))
        for i in range(1, len(self.vocab)):
            word_context.fill(i)
            similarities[:,i] = self.validation_model.predict_on_batch([word_target, word_context])
        return similarities

    def _find_similar_words(self, epoch):
        #if (epoch+1) % 5 != 0: return

        similarities = self._get_token_similarities(self.validation_tokens)
        sorted_tokens = similarities.argsort(axis=1)
        top_k = sorted_tokens[:,-1:-2-NUM_SIMILAR_WORDS:-1]
        print(self.vocab[top_k])

def train(output_path="weights/embedding"):
    tokenized_data, vocab = input.tokenized()
    vocab = np.array(vocab)

    validation_tokens = list(range(24,4*16+1,4))#random.sample(range(1,len(vocab)), 16)


    # build negative sampling skipgrams
    sampling_table = keras.preprocessing.sequence.make_sampling_table(len(vocab))
    word_pairs, labels = keras.preprocessing.sequence.skipgrams(tokenized_data,
            len(vocab),
            sampling_table=sampling_table,
            window_size=SKIP_WINDOW,
            shuffle=True)

    # convert to numpy arrays
    # reshape pairs from (?,2) to ((?,1), (?,1)), labels from (?) to (?,1)
    word_pairs = list(np.expand_dims(np.array(a, dtype=np.int16), axis=1)
            for a in zip(*word_pairs))
    """ necessary for tf.data.Dataset because nested inputs are broken
    word_pairs = np.stack(word_pairs)
    word_pairs = word_pairs.transpose((1,0,2))
    """
    labels = np.expand_dims(np.array(labels, dtype=np.int16), axis=1)

    # train the model
    train_model, validation_model = negative_sampling_model(len(vocab),
            EMBEDDING_OUTPUT_SIZE)
    train_model.compile(optimizer="rmsprop", loss="binary_crossentropy")

    """ this doesn't work with tf.keras for some reason """
    train_model.fit(x=word_pairs, y=labels,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[SimilarityCallback(vocab,
                                                  validation_tokens,
                                                  validation_model)])

    # save the learned embedding layer weights
    embedding_layer = next(embedding_layer for embedding_layer in
        train_model.layers if isinstance(embedding_layer, Embedding))
    weights = embedding_layer.get_weights()
    np.save(output_path, weights)

def load(embedding_path="weights/embedding.npy"):
    return np.load(embedding_path)

""" the way tf.data.Dataset *should* train
data = (tf.data.Dataset.from_tensor_slices((word_pairs, labels))
        .batch(BATCH_SIZE)
        .repeat())

train_model.fit(data,
                steps_per_epoch=STEPS_PER_EPOCH,
                epochs=EPOCHS,
                callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=_find_similar_words)])

"""

""" the way you actually have to train it
next_batch = data.make_one_shot_iterator().get_next()
for i in range(EPOCHS):
    print("Epoch {}/{}".format(i,EPOCHS))
    for j in range(STEPS_PER_EPOCH):
        batch = sess.run(next_batch)
        history = train_model.fit(x=batch[0],y=batch[1],batch_size=BATCH_SIZE, verbose=0)
        print("\r{}/{} {}".format(j,STEPS_PER_EPOCH,history.history), end="")
    print()
    _find_similar_words(i)
"""

if __name__ == "__main__":
    train()
