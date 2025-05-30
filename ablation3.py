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
import tensorflow as tf

# -------------------- Load and preprocess the dataset --------------------
df = pd.read_csv("ds3.csv")  # Update path if needed
X_text = df["Query"].astype(str)
y = df["Label"].values

# Tokenization
VOCAB_SIZE = 10000
MAX_LEN = 100
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
X_seq = tokenizer.texts_to_sequences(X_text)
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, stratify=y, random_state=42)

# -------------------- Define the SQLGuardNet model --------------------
def build_sqlguardnet_ablation(use_transformer=True, use_cnn=True, use_lstm=True,
                                vocab_size=VOCAB_SIZE, max_len=MAX_LEN, embed_dim=128,
                                num_heads=4, d_conv=64, d_lstm=64, dense_units=128):
    input_layer = Input(shape=(max_len,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len)
    x_embed = embedding(input_layer)

    branches = []

    # Transformer Branch
    if use_transformer:
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x_embed, x_embed)
        trans_norm = LayerNormalization(epsilon=1e-6)(Lambda(lambda x: x)(attn))
        trans_pool = GlobalMaxPooling1D()(trans_norm)
        branches.append(trans_pool)

    # CNN Branch
    if use_cnn:
        conv = Conv1D(filters=d_conv, kernel_size=5, activation='relu')(x_embed)
        maxpool = MaxPooling1D(pool_size=5)(conv)
        cnn_pool = GlobalMaxPooling1D()(maxpool)
        branches.append(cnn_pool)

    # LSTM Branch
    if use_lstm:
        lstm_out = LSTM(d_lstm)(x_embed)
        branches.append(lstm_out)

    # Concatenate active branches
    if len(branches) == 1:
        x = branches[0]
    else:
        x = Concatenate()(branches)

    # Dense layers
    dense1 = Dense(dense_units, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# -------------------- Define ablation configurations --------------------
ablation_configs = {
    "Full Model": (True, True, True),
    "No Transformer": (False, True, True),
    "No CNN": (True, False, True),
    "No LSTM": (True, True, False),
    "Only Transformer": (True, False, False),
    "Only CNN": (False, True, False),
    "Only LSTM": (False, False, True)
}

# -------------------- Train and Evaluate Each Configuration --------------------
results = {}
es = EarlyStopping(patience=3, restore_best_weights=True)

for label, (use_transformer, use_cnn, use_lstm) in ablation_configs.items():
    print(f"\nTraining configuration: {label}")
    model = build_sqlguardnet_ablation(use_transformer, use_cnn, use_lstm)
    model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=64, callbacks=[es], verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report["accuracy"] * 100
    results[label] = accuracy
    print(f"Test Accuracy for {label}: {accuracy:.2f}%")

# -------------------- Final Results --------------------
print("\nAblation Study Results:")
for model, acc in results.items():
    print(f"{model}: {acc:.2f}%")
