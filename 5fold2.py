import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, MultiHeadAttention, LayerNormalization, 
                                     Lambda, Conv1D, MaxPooling1D, GlobalMaxPooling1D, 
                                     LSTM, Dense, Concatenate)

# Load dataset
df = pd.read_csv('ds2.csv')
X_text = df['Query'].astype(str)
y = df['Label'].values

# Parameters
MAX_LEN = 100
VOCAB_SIZE = 10000
EMBED_DIM = 128
NUM_HEADS = 4
D_CONV = 128
D_LSTM = 128
DENSE_UNITS = 128
EPOCHS = 5
BATCH_SIZE = 64
N_SPLITS = 5

# Tokenize and pad
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
X_seq = tokenizer.texts_to_sequences(X_text)
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Define model builder
def build_model():
    input_layer = Input(shape=(MAX_LEN,))
    embedding_layer = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN)
    
    x_embed = embedding_layer(input_layer)
    
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
    
    # Merge and Dense
    concat = Concatenate()([trans_pool, cnn_pool, lstm_out])
    dense1 = Dense(DENSE_UNITS, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense1)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_pad, y), 1):
    print(f"\nFold {fold}")
    X_train, X_test = X_pad[train_idx], X_pad[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = build_model()
    es = EarlyStopping(patience=2, restore_best_weights=True)
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)
    
    y_pred = model.predict(X_test) > 0.5
    acc = accuracy_score(y_test, y_pred)
    fold_accuracies.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")

# Overall performance
print("\nAverage Accuracy:", np.mean(fold_accuracies))
