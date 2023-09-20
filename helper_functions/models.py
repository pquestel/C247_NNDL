import tensorflow.nn as nn
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

def eeg_CNN_LSTM_Model(input_shape, show_summary = False):
  ''' CNN-LSTM Model
  2 layer CNN followed by two stacked LSTM layerx
  '''

  FILT_LEN = 10
  CONV_DROPOUT = 0.5
  LSTM_DROPOUT = 0.5

  # Input
  inputs = keras.Input(shape=input_shape, name="Filterd_EEG")

  # Block 1
  x = layers.Conv2D(25, (FILT_LEN, 1), strides=(1,1), activation='relu', name="conv1D1", padding = 'same')(inputs)
  x = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maxpool1')(x)
  x = layers.BatchNormalization(axis=-1, name = 'BN1')(x)
  x = keras.layers.Dropout(CONV_DROPOUT, name = 'drop1')(x)

  # Block 2
  x = layers.Conv2D(50, (FILT_LEN, 1), strides=(1,1), name="conv1D2", activation='relu', padding='same')(x)
  x = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maxpool2')(x)
  x = layers.BatchNormalization(axis=-1, name = 'BN2')(x)
  x = keras.layers.Dropout(CONV_DROPOUT, name = 'drop2')(x)

  # LSTM
  x = layers.Reshape((x.shape[1],x.shape[3]))(x)
  x = layers.LSTM(32, return_sequences=True, dropout=LSTM_DROPOUT, name="LSTM-1")(x)
  x = layers.LSTM(32, return_sequences=False, dropout=LSTM_DROPOUT, name="LSTM-2")(x)

  # Softmax
  class_probs = layers.Dense(4, activation=nn.softmax, name="class-probs")(x)

  cnn_lstm_clf = keras.Model(inputs=inputs, outputs=class_probs, name="eeg_cnn")

  if show_summary:
    cnn_lstm_clf.summary()

  return cnn_lstm_clf

def eeg_CNN_FFT_Model(input_shape, show_summary = False):
  # hyperparameters
  CONV_DROPOUT = 0.5
  FILT_LEN = 10

  inputs_time = keras.Input(shape=input_shape, name="time")
  inputs_fft = keras.Input(shape=input_shape, name="fft")

  # Block 1
  x = layers.Conv2D(25, (FILT_LEN, 1), strides=(1,1), activation='relu', name="conv1D1", padding = 'same')(inputs_time)
  x = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maxpool1')(x)
  x = layers.BatchNormalization(axis=-1, name = 'BN1')(x)
  x = keras.layers.Dropout(CONV_DROPOUT, name = 'drop1')(x)

  # Block 2
  x = layers.Conv2D(50, (FILT_LEN, 1), strides=(1,1), name="conv1D2", activation='relu', padding='same')(x)
  x = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maxpool2')(x)
  x = layers.BatchNormalization(axis=-1, name = 'BN2')(x)
  x = keras.layers.Dropout(CONV_DROPOUT, name = 'drop2')(x)

  # Block 3
  x = layers.Conv2D(100, (FILT_LEN, 1), strides=(1,1), name="conv1D3", activation='relu', padding='same')(x)
  x = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maxpool3')(x)
  x = layers.BatchNormalization(axis=-1, name = 'BN3')(x)
  x = keras.layers.Dropout(CONV_DROPOUT, name = 'drop3')(x)

  # Reshape
  time_output = layers.Flatten(name = 'Flatten')(x)


  # Block 1
  y = layers.Conv2D(25, (FILT_LEN, 1), strides=(1,1), activation='relu', name="conv1D1_fft", padding = 'same')(inputs_fft)
  y = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maypool1')(y)
  y = layers.BatchNormalization(axis=-1, name = 'BN1_fft')(y)
  y = keras.layers.Dropout(CONV_DROPOUT, name = 'drop1_fft')(y)

  # Block 2
  y = layers.Conv2D(50, (FILT_LEN, 1), strides=(1,1), name="conv1D2_fft", activation='relu', padding='same')(y)
  y = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maypool2')(y)
  y = layers.BatchNormalization(axis=-1, name = 'BN2_fft')(y)
  y = keras.layers.Dropout(CONV_DROPOUT, name = 'drop2_fft')(y)

  # Block 3
  y = layers.Conv2D(100, (FILT_LEN, 1), strides=(1,1), name="conv1D3_fft", activation='relu', padding='same')(y)
  y = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maypool3')(y)
  y = layers.BatchNormalization(axis=-1, name = 'BN3_fft')(y)
  y = keras.layers.Dropout(CONV_DROPOUT, name = 'drop3_fft')(y)

  # Reshape
  fft_output = layers.Flatten(name = 'Flatten_fft')(y)

  '''
  ### fft input block
  y = layers.Conv2D(40, (200, 1), strides = (1,1), name='conv_fft', activation='relu')(inputs_fft)
  y = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maxpool2_fft')(y)
  y = keras.layers.Dropout(CONV_DROPOUT, name = 'drop1_fft')(y)
  y = layers.Flatten(name = 'Flatten_fft')(y)
  y = layers.Dense(900, activation='relu', name="fully-connected-fft")(y)
  y = layers.Dropout(DENSE_DROPOUT, name = 'dense_drop2')(y)
  fft_output = layers.BatchNormalization(axis=-1, epsilon=batchnorm_eps, momentum=batchnorm_momentum, name='bn_fft')(y) * 0
  '''

  # add together
  z = keras.layers.add([time_output, fft_output])

  # Softmax
  class_probs = layers.Dense(4, activation=nn.softmax, name="class-probs")(z)

  # Model
  model = keras.Model(inputs=[inputs_time, inputs_fft], outputs=class_probs, name="m1")
  return model

def eeg_CNN_Model(input_shape, show_summary = False):
  ''' Default params scores testing 70+ for both 1 and all subjects
  '''

  FILT_LEN = 10
  CONV_DROPOUT = 0.60
  
  # Input
  inputs = keras.Input(shape=input_shape, name="Filterd_EEG")

  # Block 1
  x = layers.Conv2D(25, (FILT_LEN, 1), strides=(1,1), activation='relu', name="conv1D1", padding = 'same')(inputs)
  x = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maxpool1')(x)
  x = layers.BatchNormalization(axis=-1, name = 'BN1')(x)
  x = keras.layers.Dropout(CONV_DROPOUT, name = 'drop1')(x)

  # Block 2
  x = layers.Conv2D(50, (FILT_LEN, 1), strides=(1,1), name="conv1D2", activation='relu', padding='same')(x)
  x = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maxpool2')(x)
  x = layers.BatchNormalization(axis=-1, name = 'BN2')(x)
  x = keras.layers.Dropout(CONV_DROPOUT, name = 'drop2')(x)

  # Block 3
  x = layers.Conv2D(100, (FILT_LEN, 1), strides=(1,1), name="conv1D3", activation='relu', padding='same')(x)
  x = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name = 'maxpool3')(x)
  x = layers.BatchNormalization(axis=-1, name = 'BN3')(x)
  x = keras.layers.Dropout(CONV_DROPOUT, name = 'drop3')(x)

  # Reshape
  x = layers.Flatten(name = 'Flatten')(x)

  # Softmax
  class_probs = layers.Dense(4, activation=nn.softmax, name="class-probs")(x)

  cnn_clf = keras.Model(inputs=inputs, outputs=class_probs, name="eeg_cnn")

  if show_summary:
    cnn_clf.summary()

  return cnn_clf

def transformer_model(input_shape, show_summary = False):

  # build model

  model = build_model(
    input_shape,
    head_size=40,
    num_heads=1,
    ff_dim=25,
    num_transformer_blocks=1,
    mlp_units=[256],
    mlp_dropout=0.5,
    dropout=0.3
  )
  return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    #x = layers.Conv1D(filters=ff_dim, kernel_size=(1), activation="relu",activity_regularizer=tf.keras.regularizers.L2(0.001))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=(1),activity_regularizer=tf.keras.regularizers.L2(0.001))(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    ):
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(4, activation="softmax")(x)


    return keras.Model(inputs, outputs)