import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, MultiHeadAttention, LayerNormalization, Lambda,
    SeparableConv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM,
    Dense, Bidirectional, Multiply, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load dataset
df = pd.read_csv("ds2.csv")
X_text = df["Query"].astype(str)
y = df["Label"].values

# Preprocessing
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBED_DIM = 128

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)

X_seq = tokenizer.texts_to_sequences(X_text)
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, stratify=y, random_state=42)

# Build model with weighted concatenation
def build_sqlguardnet_weighted():
    NUM_HEADS = 4
    D_CONV = 64
    D_LSTM = 64
    DENSE_UNITS = 128

    input_layer = Input(shape=(MAX_LEN,))
    x_embed = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN)(input_layer)

    # Transformer branch
    attn = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM)(x_embed, x_embed)
    trans_norm = LayerNormalization(epsilon=1e-6)(attn)
    trans_pool = GlobalMaxPooling1D()(trans_norm)

    # CNN branch
    conv = SeparableConv1D(filters=D_CONV, kernel_size=5, activation='relu')(x_embed)
    maxpool = MaxPooling1D(pool_size=5)(conv)
    cnn_pool = GlobalMaxPooling1D()(maxpool)

    # BiLSTM branch
    lstm_out = Bidirectional(LSTM(D_LSTM, return_sequences=False))(x_embed)

    # Learnable scalar weights for each branch
    w1 = tf.Variable(1.0, trainable=True, name='w_transformer', dtype=tf.float32)
    w2 = tf.Variable(1.0, trainable=True, name='w_cnn', dtype=tf.float32)
    w3 = tf.Variable(1.0, trainable=True, name='w_lstm', dtype=tf.float32)

    # Weighted sum
    weighted_trans = Multiply()([trans_pool, w1])
    weighted_cnn = Multiply()([cnn_pool, w2])
    weighted_lstm = Multiply()([lstm_out, w3])

    weighted_merge = Add()([weighted_trans, weighted_cnn, weighted_lstm])

    # Classification layers
    dense1 = Dense(DENSE_UNITS, activation='relu')(weighted_merge)
    output = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Train and evaluate
model = build_sqlguardnet_weighted()
es = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=64, callbacks=[es])

# Throughput + Evaluation
start_time = time.time()
y_pred_prob = model.predict(X_test, batch_size=64)
end_time = time.time()

y_pred = (y_pred_prob > 0.5).astype(int)
throughput = len(X_test) / (end_time - start_time)

print(f"\nThroughput: {throughput:.2f} predictions/sec")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
