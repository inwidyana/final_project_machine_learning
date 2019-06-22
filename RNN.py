from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense, Layer
import numpy as np

############################################################################################################
# Variable init
############################################################################################################

batch_size = 64  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 100  # Number of samples to train on. My memory is killing me

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

input_texts = input_texts[: min(num_samples, len(input_texts) - 1)]
target_texts = target_texts[: min(num_samples, len(target_texts) - 1)]

# Add new chars
for line in input_texts:
    for char in line:
        if char not in input_characters:
            input_characters.add(char)
for line in target_texts:
    for char in line:
        if char not in target_characters:
            target_characters.add(char)

# Sort
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

# How many different characters
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

# Sequence max lengths
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of training data :', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max input length:', max_encoder_seq_length)
print('Max output length:', max_decoder_seq_length)

# Map a number to the characters
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# (size of input, max sequence, number of unique possible values)
encoder_input = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens))
decoder_input = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens))
decoder_target = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens))

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

class MinimalRNNCell(Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer='uniform',
                                        name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = np.dot(inputs, self.kernel)
        output = h + np.dot(prev_output, self.recurrent_kernel)
        return output, [output]

cell = MinimalRNNCell(latent_dim)

# Initiate Keras tensor input
encoder_inputs = Input(shape=(None, num_encoder_tokens))

# Return state = output of the last time step , return sequence = returns all time step
encoder = SimpleRNN(latent_dim, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs)

############################################################################################################
# Decoder
############################################################################################################

cell = MinimalRNNCell(latent_dim)

# Set up the decoder, using encoder_states as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_rnn = SimpleRNN(latent_dim, return_sequences=True)
decoder_outputs = decoder_rnn(decoder_inputs,
                                     initial_state=state_h)

# Softmax layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model architecture
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Train
model.compile(optimizer ='rmsprop', loss ='categorical_crossentropy', metrics = ['accuracy'])
model.fit([encoder_input, decoder_input], decoder_target,
          batch_size=batch_size,
          epochs=epochs,
validation_split=0.2)

model.save('s2s.h5')