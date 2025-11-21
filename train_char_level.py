#!/usr/bin/env python3
"""
Character-level Bi-LSTM for Shakespeare text generation.
Optimized for 70%+ validation accuracy.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os

# Configuration
SEQUENCE_LENGTH = 100  # Characters to look back
BATCH_SIZE = 512  # Large batches for speed
EMBEDDING_DIM = 128
LSTM_UNITS = 256
EPOCHS = 30
VALIDATION_SPLIT = 0.15

print("="*60)
print("CHARACTER-LEVEL BI-LSTM SHAKESPEARE TEXT GENERATION")
print("="*60)

# Load data
print("\n📖 Loading Shakespeare text...")
with open('Shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total characters: {len(text):,}")

# Create character mappings
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

print(f"Unique characters (vocab): {vocab_size}")
print(f"Sample characters: {chars[:20]}")

# Prepare sequences
print("\n🔄 Creating training sequences...")
X = []
y = []

for i in range(0, len(text) - SEQUENCE_LENGTH, 3):  # Step of 3 for speed
    sequence = text[i:i + SEQUENCE_LENGTH]
    target = text[i + SEQUENCE_LENGTH]
    X.append([char_to_idx[c] for c in sequence])
    y.append(char_to_idx[target])

X = np.array(X)
y = np.array(y)

print(f"Training sequences: {len(X):,}")
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")

# Split data
split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"\n📊 Data split:")
print(f"  Training: {len(X_train):,} sequences")
print(f"  Validation: {len(X_val):,} sequences")

# Build model
print("\n🏗️  Building Bi-LSTM model...")
model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2)),
    Bidirectional(LSTM(LSTM_UNITS // 2, dropout=0.2)),
    Dropout(0.3),
    Dense(LSTM_UNITS, activation='relu'),
    Dropout(0.3),
    Dense(vocab_size, activation='softmax')
], name='CharLevel_BiLSTM')

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.002),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')]
)

print("\n📋 Model Summary:")
model.summary()

# Callbacks
os.makedirs('models', exist_ok=True)
os.makedirs('logs/char_level', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'models/char_bilstm_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
]

# Train
print("\n🚀 Starting training...")
print("="*60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Final evaluation
print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60)

train_loss, train_acc, train_top5 = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
val_loss, val_acc, val_top5 = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)

print(f"\n✅ Training Metrics:")
print(f"   Accuracy: {train_acc*100:.2f}%")
print(f"   Top-5 Accuracy: {train_top5*100:.2f}%")
print(f"   Loss: {train_loss:.4f}")

print(f"\n🎯 Validation Metrics:")
print(f"   Accuracy: {val_acc*100:.2f}%")
print(f"   Top-5 Accuracy: {val_top5*100:.2f}%")
print(f"   Loss: {val_loss:.4f}")

# Generate sample text
print("\n📝 Sample text generation:")
print("-"*60)

def generate_text(model, seed_text, length=200):
    """Generate text using the trained model"""
    generated = seed_text

    for _ in range(length):
        # Prepare input
        x = np.array([[char_to_idx.get(c, 0) for c in generated[-SEQUENCE_LENGTH:]]])

        # Predict next character
        predictions = model.predict(x, verbose=0)[0]
        next_idx = np.argmax(predictions)
        next_char = idx_to_char[next_idx]

        generated += next_char

    return generated

seed = "To be or not to be"
generated_text = generate_text(model, seed, 200)
print(generated_text)
print("-"*60)

print("\n✨ Training complete!")
print(f"Best model saved to: models/char_bilstm_best.h5")
print("="*60)

# Save char mappings
import pickle
with open('models/char_mappings.pkl', 'wb') as f:
    pickle.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)
print("Character mappings saved to: models/char_mappings.pkl")
