import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, LSTM, GRU, Dense, Dropout, BatchNormalization, Input, Reshape

def build_baseline_lstm(input_shape, horizon=5, units=128, dropout=0.3):
    inputs = Input(shape=input_shape)
    x = LSTM(units, return_sequences=False)(inputs)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(horizon * 4)(x)
    outputs = Reshape((horizon, 4))(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Baseline_LSTM")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_baseline_gru(input_shape, horizon=5, units=128, dropout=0.3):
    inputs = Input(shape=input_shape)
    x = GRU(units, return_sequences=False)(inputs)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(horizon * 4)(x)
    outputs = Reshape((horizon, 4))(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Baseline_GRU")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_baseline_tcn(input_shape, horizon=5, filters=64, dilations=[1,2,4,8], dropout=0.3):
    inputs = Input(shape=input_shape)
    x = inputs
    for idx, d in enumerate(dilations):
        x = Conv1D(filters=filters, kernel_size=3, padding='causal', dilation_rate=d, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(horizon * 4)(x)
    outputs = Reshape((horizon, 4))(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Baseline_TCN")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
