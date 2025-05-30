# Full Implementation of Multiple Models for SQL Query Classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Embedding, LSTM, Dense, Conv1D, MaxPooling1D,
                                     GlobalMaxPooling1D, Flatten, Dropout, Concatenate, LayerNormalization,
                                     MultiHeadAttention, Lambda)
from tensorflow.keras.applications import ResNet50, ResNet50V2, DenseNet121, DenseNet169, InceptionV3, VGG16, VGG19
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load dataset
df = pd.read_csv("ds1.csv")
X_text = df["Query"].astype(str)
y = df["Label"].values

# Tokenization and padding
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBED_DIM = 128
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
X_seq = tokenizer.texts_to_sequences(X_text)
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, stratify=y, random_state=42)

# Embedding Layer
def embedding_layer(input_layer):
    return Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN)(input_layer)

# SQLGuardNet model
def build_sqlguardnet():
    input_layer = Input(shape=(MAX_LEN,))
    x_embed = embedding_layer(input_layer)

    # Transformer Branch
    attn = MultiHeadAttention(num_heads=4, key_dim=EMBED_DIM)(x_embed, x_embed)
    trans_norm = LayerNormalization(epsilon=1e-6)(Lambda(lambda x: x)(attn))
    trans_pool = GlobalMaxPooling1D()(trans_norm)

    # CNN Branch
    conv = Conv1D(filters=64, kernel_size=5, activation='relu')(x_embed)
    maxpool = MaxPooling1D(pool_size=5)(conv)
    cnn_pool = GlobalMaxPooling1D()(maxpool)

    # LSTM Branch
    lstm_out = LSTM(64)(x_embed)

    # Concatenate
    concat = Concatenate()([trans_pool, cnn_pool, lstm_out])
    dense = Dense(128, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Simple LSTM model
def build_lstm():
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        embedding_layer(Input(shape=(MAX_LEN,))),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Simple CNN model
def build_cnn():
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        embedding_layer(Input(shape=(MAX_LEN,))),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(pool_size=2),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Transformer-only
def build_transformer_only():
    input_layer = Input(shape=(MAX_LEN,))
    x_embed = embedding_layer(input_layer)
    attn = MultiHeadAttention(num_heads=4, key_dim=EMBED_DIM)(x_embed, x_embed)
    norm = LayerNormalization(epsilon=1e-6)(attn)
    pool = GlobalMaxPooling1D()(norm)
    output = Dense(1, activation='sigmoid')(pool)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrapper for pretrained models
pretrained_models = {
    "ResNet50": ResNet50,
    "ResNet50V2": ResNet50V2,
    "DenseNet121": DenseNet121,
    "DenseNet169": DenseNet169,
    "InceptionV3": InceptionV3,
    "VGG16": VGG16,
    "VGG19": VGG19
}

def build_pretrained_model(base_model_class, name):
    input_layer = Input(shape=(MAX_LEN,))
    x_embed = embedding_layer(input_layer)
    x = tf.expand_dims(x_embed, -1)
    x = tf.keras.layers.Reshape((MAX_LEN, EMBED_DIM, 1))(x)
    x = tf.image.resize(x, (224, 224))

    base_model = base_model_class(weights=None, include_top=False, input_shape=INPUT_SHAPE)
    x = base_model(x)
    x = GlobalMaxPooling2D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Model list
models = {
    "SQLGuardNet": build_sqlguardnet(),
    "Simple LSTM": build_lstm(),
    "1D CNN": build_cnn(),
    "Transformer-only": build_transformer_only(),
}

# Add pretrained CNNs
for name, model_fn in pretrained_models.items():
    try:
        models[name] = build_pretrained_model(model_fn, name)
    except Exception as e:
        print(f"Skipping {name} due to error: {e}")

# Training and evaluation loop
for name, model in models.items():
    print(f"\nTraining model: {name}")
    es = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=64, callbacks=[es], verbose=1)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(f"Results for {name}:")
    print(classification_report(y_test, y_pred))
