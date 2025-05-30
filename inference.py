import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Embedding, LSTM, Dense, Conv1D,
                                     MaxPooling1D, GlobalMaxPooling1D,
                                     MultiHeadAttention, LayerNormalization, Concatenate)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load dataset
df = pd.read_csv("ds1.csv")
texts = df["Query"].astype(str).tolist()

# 2. Tokenize and pad
VOCAB_SIZE = 10000
MAX_LEN = 100
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X_pad = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
X_sample = X_pad[:100]  # sample for measuring inference time

# 3. Inference time function
def measure_inference_time(model, X, name):
    start = time.time()
    _ = model.predict(X, verbose=0)
    duration = time.time() - start
    avg_ms = (duration / len(X)) * 1000
    return (name, avg_ms)

# 4. Build baseline models
def build_simple_lstm():
    model = Sequential([
        Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_simple_cnn():
    model = Sequential([
        Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(pool_size=2),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_transformer_only():
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(VOCAB_SIZE, 64)(inputs)
    x = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = LayerNormalization()(x)
    x = GlobalMaxPooling1D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def build_sqlguardnet():
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(VOCAB_SIZE, 64)(inputs)

    # Transformer branch
    t = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    t = LayerNormalization()(t)
    t = GlobalMaxPooling1D()(t)

    # CNN branch
    c = Conv1D(64, 5, activation='relu')(x)
    c = MaxPooling1D(2)(c)
    c = GlobalMaxPooling1D()(c)

    # LSTM branch
    l = LSTM(64)(x)

    concat = Concatenate()([t, c, l])
    dense = Dense(128, activation='relu')(concat)
    outputs = Dense(1, activation='sigmoid')(dense)

    return Model(inputs, outputs)

# 5. Define models to evaluate
models = {
    "SQLGuardNet": build_sqlguardnet(),
    "Simple LSTM": build_simple_lstm(),
    "1D CNN": build_simple_cnn(),
    "Transformer-only": build_transformer_only()
}

# Compile each model
for m in models.values():
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 6. Measure inference time
print("Model\t\t\tInference Time (ms)")
for name, model in models.items():
    model.build(input_shape=(None, MAX_LEN))
    model.summary()
    model_name, ms = measure_inference_time(model, X_sample, name)
    print(f"{model_name:20s}: {ms:.2f} ms")
