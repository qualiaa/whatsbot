#!/usr/bin/env python3
import sys
import numpy as np
import tensorflow as tf

vocab = []
with open("data/vocab.txt") as f:
    for line in f.readlines():
        try:
            vocab.append(line.split()[1])
        except:
            print(line)
            sys.exit(1)

vocab.append("UNKNOWN")
    
with open("data/network_input.txt") as f:
    all_data=[int(token) for token in f.readlines()]

def batch_dataset(all_data, examples_per_batch, words_per_example):
    num_words = len(all_data)

    words_per_batch = examples_per_batch * words_per_example
    
    num_batches = num_words // words_per_batch
    num_examples = examples_per_batch * num_batches
    num_words = (words_per_batch * num_batches) + 1 # drop extra words
    print(num_batches)
    print(num_examples)
    print(num_words)

    batch_stride = num_batches

    words = tf.data.Dataset.from_tensor_slices(all_data[:num_words])
    examples = (tf.data.Dataset.zip((words,words.skip(1)))
            .batch(words_per_example))
    
    example_shards = examples.batch(batch_stride)
    
    strided_examples = example_shards.interleave(
            lambda *a: tf.data.Dataset.from_tensor_slices(a),
            cycle_length=examples_per_batch)

    batches = (strided_examples
            .batch(examples_per_batch))

    return batches

batches = batch_dataset(all_data, 100, 20)
dataset_batch = batches.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    try:
        while True:
            a = sess.run(dataset_batch)
    except tf.errors.OutOfRangeError: pass

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

