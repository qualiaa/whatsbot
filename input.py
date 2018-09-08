#!/usr/bin/env python3

import sys
import functools

import tensorflow as tf


def tokenized():
    # keras assumes that first token is unknown
    vocab = ["UNKNOWN"]
    with open("data/vocab.txt") as f:
        try:
            vocab.extend(line.split()[1] for line in f.readlines())
        except:
            print("Could not split line")
            sys.exit(1)

    with open("data/network_input.txt") as f:
        tokenized_data=[int(token) for token in f.readlines()]

    return tokenized_data, vocab

def batch_dataset(dataset, examples_per_batch, timesteps_per_example, num_timesteps):
    timesteps_per_batch = examples_per_batch * timesteps_per_example

    num_batches = num_timesteps // timesteps_per_batch
    num_examples = examples_per_batch * num_batches
    num_timesteps = (timesteps_per_batch * num_batches) + 1 # drop extra timesteps

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

"""
dataset = tf.data.Dataset.from_tensor_slices(raw_data)
batches = dataset.apply(
        to_rnn_input(examples_per_batch=100,
                     timesteps_per_example=20,
                     num_timesteps=len(raw_data)))

dataset_batch = batches.make_one_shot_iterator().get_next()
"""
