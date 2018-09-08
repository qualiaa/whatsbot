#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.data import Dataset
from numpy import array



num_words = 10000
words_per_example = 20
examples_per_batch = 100

words_per_batch = examples_per_batch * words_per_example

num_batches = num_words // words_per_batch
num_words =  num_batches * words_per_batch
num_examples = num_words // words_per_example

a = Dataset.range(num_words).batch(words_per_example)
batch_stride = num_batches

a = a.batch(batch_stride)
a = a.interleave(lambda a: Dataset.from_tensor_slices(a),
        cycle_length=examples_per_batch)

a = a.batch(examples_per_batch)
print(a.output_shapes)

n = a.make_one_shot_iterator().get_next()


print("-"*80)
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(n))
    except tf.errors.OutOfRangeError:
        pass
