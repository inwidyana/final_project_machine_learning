from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import re
import sys
np.set_printoptions(threshold=sys.maxsize)

############################################################################################################
# Variable init
############################################################################################################

batch_size = 64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 20  # Number of samples to train on. My memory is killing me

# Dataset
data_de = 'training/news-commentary-v8.de-en.de'
data_en = 'training/news-commentary-v8.de-en.en'

############################################################################################################
# Vectorize the data.
############################################################################################################

# Set contains unique values
input_characters = set()
target_characters = set()

# Split each line to lists
with open(data_de, 'r', encoding='utf-8') as data:
    input_texts = data.read().split('\n')
with open(data_en, 'r', encoding='utf-8') as data:
    target_texts = data.read().split('\n')

# Limit to number of samples
input_texts = input_texts[: min(num_samples, len(input_texts) - 1)]
target_texts = target_texts[: min(num_samples, len(target_texts) - 1)]

# Add new unique words
for line in input_texts:
    words = line.split(" ")
    for words in words:
        if word not in input_words:
            input_words.add(word)

for line in target_texts:
    words = line.split(" ")
    for words in words:
        if word not in target_words:
            target_words.add(word)

# Sort
input_words = sorted(list(input_words))
target_words = sorted(list(target_words))

# How many different characters
num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)

# Sequence max lengths
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# Dataset details display
print('Number of training data :', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max input length:', max_encoder_seq_length)
print('Max output length:', max_decoder_seq_length)

# Map a number to the characters (character, index)
input_map = dict([(char, i) for i, char in enumerate(input_words)])
target_map = dict([(char, i) for i, char in enumerate(target_words)])

# (size of input, max sequence, number of unique possible values)
encoder_input_template = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens))
decoder_input_template = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens))
decoder_target_template = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens))

# zip returns set of index + item
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep and will not include the start character.
            decoder_target[i, t - 1, target_token_index[char]] = 1.

############################################################################################################
# Encoder
############################################################################################################

# Initiate Keras tensor input
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# Return state = output of the last time step , return sequence = returns all time step
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states. h = hidden, c = cell
encoder_states = [state_h, state_c]

############################################################################################################
# Decoder
############################################################################################################

# Set up the decoder, using encoder_states as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

# Softmax layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model architecture
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Train
model.compile(optimizer ='Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit([encoder_input, decoder_input], decoder_target,
          batch_size=batch_size,
          epochs=epochs,
validation_split=0.1)

model.save('s2s.h5')

# Inference
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        print(sampled_token_index)
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
            len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

        return decoded_sentence


for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)