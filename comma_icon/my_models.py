# models
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Conv1D,
    MaxPooling1D,
    Dropout,
    Dense,
)
from tensorflow.keras import Model
import config

# import tensorflow as tf


def build_model_1():
    config.set_seed()
    main_input = Input(shape=(config.MAX_SEQ_LEN,), dtype="int32", name="main_input")
    x = Embedding(
        input_dim=config.VOCAB_SIZE, output_dim=50, input_length=config.MAX_SEQ_LEN
    )(main_input)
    x = Dropout(0.3)(x)
    x = Conv1D(64, 5, activation="relu")(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = LSTM(100)(x)
    x = Dropout(0.3)(x)

    output_array = [
        Dense(3, activation="softmax", name="taskA_op")(x),
        Dense(1, activation="sigmoid", name="taskB_op")(x),
        Dense(1, activation="sigmoid", name="taskC_op")(x),
    ]

    model = Model(inputs=main_input, outputs=output_array)
    return model
