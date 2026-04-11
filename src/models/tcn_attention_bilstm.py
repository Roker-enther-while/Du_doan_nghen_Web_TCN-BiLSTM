import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
try:
    from .attention_layer import FeatureAttention, TemporalAttention
except ImportError:
    from attention_layer import FeatureAttention, TemporalAttention

def build_advanced_model(input_shape, horizon=5, num_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], lstm_units=128, dropout_rate=0.3):
    """
    V3 SOTA Architecture: Dual-Stage Attention TCN-BiLSTM
    - Stage 1: Feature Attention (Input weighting)
    - TCN Block: Filters=64, Dilations=[1, 2, 4, 8]
    - BiLSTM Block: 128 Units
    - Stage 2: Temporal Attention (Sequence weighting)
    - Output: Multi-variate MIMO prediction
    """
    inputs = Input(shape=input_shape)

    # 1. Feature Attention (Stage 1)
    x = FeatureAttention(name="feature_attention")(inputs)

    # 2. TCN Block
    for idx, dilation_rate in enumerate(dilations):
        x = Conv1D(filters=num_filters,
                   kernel_size=kernel_size,
                   padding='causal',
                   activation='relu',
                   dilation_rate=dilation_rate,
                   name=f"tcn_conv_{idx+1}")(x)
        x = BatchNormalization(name=f"tcn_bn_{idx+1}")(x)
        x = Dropout(dropout_rate, name=f"tcn_drop_{idx+1}")(x)

    # 3. BiLSTM Block
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True), name="bilstm_layer")(x)
    x = Dropout(dropout_rate, name="bilstm_drop")(x)

    # 4. Temporal Attention (Stage 2)
    x = TemporalAttention(name="temporal_attention")(x)

    # 5. Dense Layers -> Output
    x = Dense(units=64, activation='relu', name="dense_1")(x)
    x = Dropout(dropout_rate, name="dense_drop")(x)
    
    # MIMO Output: (batch_size, horizon, 4 features)
    x = Dense(units=horizon * 4, name="output_dense_flat")(x)
    outputs = tf.keras.layers.Reshape((horizon, 4), name="output_mimo")(x)

    model = Model(inputs=inputs, outputs=outputs, name="WebTAB_v4.9_SOTA")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])

    return model

if __name__ == "__main__":
    # Test architecture
    SEQ_LEN = 60
    NUM_FEATURES = 13 # Match prepare_data_v2 (13-feature Multivariate)
    model = build_advanced_model((SEQ_LEN, NUM_FEATURES))
    model.summary()
