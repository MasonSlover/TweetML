import tensorflow as tf

import numpy as np
import os
import csv
import random

# true trains the model and false produces an output based on the training checkpoints
isTraining = false

# open (cleaned) tweet file
text = open('tweets.txt').read()
print(len(text))

# uncomment to truncate input for quicker training
# text = text[:100000]

# mapping unique characters to indicies
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# This variable controls the maximum input sequence length
seq_length = 280
examples_per_epoch = len(text) // (seq_length + 1)

# Creating training sets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# Batch & Buffer sizes
BATCH_SIZE = 32
BUFFER_SIZE = 10000

# shuffles the dataset
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

# constructs tensorflow model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

# loss fn
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)

model.compile(optimizer='adam', loss=loss)

# save training checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

if(isTraining):
    EPOCHS = 5
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
else:
    tf.train.latest_checkpoint(checkpoint_dir)
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([32, None]))



def generate_text(model, start_string):
    #max num of chars to generate
    num_generate = 250

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    #this var can be edited to change the output, lower numbers produce more predictable outputs
    temperature = .80

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

# generated text starters 
starters = ["Americans ", "U.S. ", "We must ", "Lazy Joe Biden ", "Dems ", "The ", "We "]

for i in range(0, 10):
  index = random.randint(0, len(starters) - 1)
  print("tweet ", i, ": ", generate_text(model, start_string=starters[index]), "\n")

