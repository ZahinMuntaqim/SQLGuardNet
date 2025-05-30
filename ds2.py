import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Input, Embedding, MultiHeadAttention, LayerNormalization, Lambda, 
                                     Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Dense, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv("ds2.csv")
X_text = df["Query"].astype(str)
y = df["Label"].values

# Tokenization and padding
VOCAB_SIZE = 10000
MAX_LEN = 100
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
X_seq = tokenizer.texts_to_sequences(X_text)
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, stratify=y, random_state=42)

# Define SQLGuardNet model
def build_sqlguardnet():
    EMBED_DIM = 128
    NUM_HEADS = 4
    D_CONV = 64
    D_LSTM = 64
    DENSE_UNITS = 128

    input_layer = Input(shape=(MAX_LEN,))
    embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN)

    x_embed = embedding(input_layer)

    # Transformer Branch
    attn = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM)(x_embed, x_embed)
    trans_norm = LayerNormalization(epsilon=1e-6)(Lambda(lambda x: x)(attn))
    trans_pool = GlobalMaxPooling1D()(trans_norm)

    # CNN Branch
    conv = Conv1D(filters=D_CONV, kernel_size=5, activation='relu')(x_embed)
    maxpool = MaxPooling1D(pool_size=5)(conv)
    cnn_pool = GlobalMaxPooling1D()(maxpool)

    # LSTM Branch
    lstm_out = LSTM(D_LSTM)(x_embed)

    # Merge and classify
    concat = Concatenate()([trans_pool, cnn_pool, lstm_out])
    dense1 = Dense(DENSE_UNITS, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model
model = build_sqlguardnet()
es = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=64, callbacks=[es])

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))
