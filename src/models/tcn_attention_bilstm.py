import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
try:
    from .attention_layer import Attention
except ImportError:
    from attention_layer import Attention

def build_advanced_model(input_shape, horizon=5, num_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], lstm_units=128, dropout_rate=0.3):
    """
    V2 Advanced Hybrid Architecture: TCN-Attention-BiLSTM
    - TCN Block: Filters=64, Dilations=[1, 2, 4, 8]
    - BiLSTM Block: 128 Units
    - Output: Multi-variate MIMO prediction (CPU, Latency, Request, Congestion_Prob)
    """
    inputs = Input(shape=input_shape)

    # 1. TCN Block
    x = inputs
    for idx, dilation_rate in enumerate(dilations):
        x = Conv1D(filters=num_filters,
                   kernel_size=kernel_size,
                   padding='causal',
                   activation='relu',
                   dilation_rate=dilation_rate,
                   name=f"tcn_conv_{idx+1}")(x)
        x = BatchNormalization(name=f"tcn_bn_{idx+1}")(x)
        x = Dropout(dropout_rate, name=f"tcn_drop_{idx+1}")(x)

    # 2. BiLSTM Block
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True), name="bilstm_layer")(x)
    x = Dropout(dropout_rate, name="bilstm_drop")(x)

    # 3. Attention Mechanism
    x = Attention(name="attention_layer")(x)

    # 4. Dense Layers -> Output
    x = Dense(units=64, activation='relu', name="dense_1")(x)
    x = Dropout(dropout_rate, name="dense_drop")(x)
    
    # MIMO Output: (batch_size, horizon, 4 features)
    # We output flat then reshape to ensure compatible keras format
    x = Dense(units=horizon * 4, name="output_dense_flat")(x)
    outputs = tf.keras.layers.Reshape((horizon, 4), name="output_mimo")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TCN_Attention_BiLSTM_V2")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])

    return model

if __name__ == "__main__":
    # Test architecture
    SEQ_LEN = 60
    NUM_FEATURES = 10 # Phase 10: 10-feature Multivariate
    model = build_advanced_model((SEQ_LEN, NUM_FEATURES))
    model.summary()
