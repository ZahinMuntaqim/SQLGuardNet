from tensorflow.keras.layers import SeparableConv1D, Bidirectional, LSTM, MultiHeadAttention, LayerNormalization, Lambda

def build_model_with_order(order):
    EMBED_DIM = 128
    NUM_HEADS = 4
    D_CONV = 64
    D_LSTM = 64
    DENSE_UNITS = 128

    input_layer = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN)(input_layer)

    for layer_name in order:
        if layer_name == "Transformer":
            x_attn = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM)(x, x)
            x = LayerNormalization(epsilon=1e-6)(x_attn)
        elif layer_name == "CNN":
            x = SeparableConv1D(filters=D_CONV, kernel_size=5, activation='relu')(x)
            x = MaxPooling1D(pool_size=5)(x)
        elif layer_name == "LSTM":
            x = Bidirectional(LSTM(D_LSTM, return_sequences=True))(x)

    x = GlobalMaxPooling1D()(x)
    x = Dense(DENSE_UNITS, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


orders = [
    ["Transformer", "CNN", "LSTM"],
    ["CNN", "Transformer", "LSTM"],
    ["Transformer", "LSTM", "CNN"],
    ["LSTM", "CNN", "Transformer"],
    ["CNN", "LSTM", "Transformer"],
    ["LSTM", "Transformer", "CNN"]
]

for i, order in enumerate(orders, 1):
    print(f"\n🔬 Training Configuration {i}: {' → '.join(order)}")
    model = build_model_with_order(order)
    es = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=64, callbacks=[es], verbose=0)

    start_time = time.time()
    y_pred_prob = model.predict(X_test, batch_size=64)
    end_time = time.time()

    y_pred = (y_pred_prob > 0.5).astype(int)
    throughput = len(X_test) / (end_time - start_time)

    print(f"🕒 Throughput: {throughput:.2f} pred/sec")
    print(classification_report(y_test, y_pred))
