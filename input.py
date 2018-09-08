#!/usr/bin/env python3
import sys
import functools

import numpy as np
import keras
import tensorflow as tf

from keras.layers import Embedding

EMBEDDING_OUTPUT_SIZE = 300

# keras assumes that first token is unknown
vocab = ["UNKNOWN"]
with open("data/vocab.txt") as f:
    try:
    vocab.extend(line.split()[1] for line in f.readlines())
    except:
        print(line)
        sys.exit(1)

with open("data/network_input.txt") as f:
    all_data=[int(token)+1 for token in f.readlines()]


def batch_dataset(dataset, examples_per_batch, timesteps_per_example, num_timesteps):
    timesteps_per_batch = examples_per_batch * timesteps_per_example

    num_batches = num_timesteps // timesteps_per_batch
    num_examples = examples_per_batch * num_batches
    num_timesteps = (timesteps_per_batch * num_batches) + 1 # drop extra timesteps
    print(num_batches)
    print(num_examples)
    print(num_timesteps)

    batch_stride = num_batches

    examples = (tf.data.Dataset.zip((dataset, dataset.skip(1)))
            .batch(timesteps_per_example, drop_remainder=True))

    example_shards = examples.batch(batch_stride, drop_remainder=True)

    strided_examples = example_shards.interleave(
            lambda *a: tf.data.Dataset.from_tensor_slices(a),
            cycle_length=examples_per_batch)

    batches = (strided_examples
            .batch(examples_per_batch, drop_remainder=True))

    return batches

def to_rnn_input(*args,**kargs):
    return functools.partial(batch_dataset,*args,**kargs)

dataset = tf.data.Dataset.from_tensor_slices(all_data)
batches = dataset.apply(
        to_rnn_input(examples_per_batch=100,
                     timesteps_per_example=20,
                     num_timesteps=len(all_data)))

dataset_batch = batches.make_one_shot_iterator().get_next()

Embedding(len(vocab), EMBEDDING_OUTPUT_SIZE)

sampling_table = keras.preprocessing.sequence.make_sampling_table(len(vocab))

"""
# old method modified from adventuresinmachinelearning
def batch_queue(data, examples_per_batch, timesteps_per_example):
    num_timesteps = len(data) - 1
    data = tf.convert_to_tensor(data, name="all_data", dtype=tf.int16)

    timesteps_per_batch = timesteps_per_example * examples_per_batch
    num_batches = num_timesteps // timesteps_per_batch
    num_timesteps = (num_batches * timesteps_per_batch)
    num_examples = num_timesteps // timesteps_per_example

    data = tf.reshape(data[:num_timesteps], [examples_per_batch, -1])

    i = tf.train.range_input_producer(num_batches, shuffle=False).dequeue()
    start_time = i*timesteps_per_example
    end_time = start_time + timesteps_per_example
    x = data[:,start_time:end_time]
    y = data[:,start_time+1:end_time+1]
    x.set_shape((examples_per_batch, timesteps_per_example))
    y.set_shape((examples_per_batch, timesteps_per_example))

    return x,y
"""

