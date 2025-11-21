#!/usr/bin/env python3
"""
Optimized Character-level Bi-LSTM for Shakespeare text generation.
Faster training while targeting 70%+ validation accuracy.

Optimizations:
- Reduced sequence length: 100 → 50
- Reduced LSTM units: 256 → 128
- Reduced embedding dim: 128 → 64
- Increased stride: 3 → 5 (fewer sequences)
- Simplified architecture for speed
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
import time

# Optimized Configuration
SEQUENCE_LENGTH = 50  # Reduced from 100
BATCH_SIZE = 512  # Keep high for CPU efficiency
EMBEDDING_DIM = 64  # Reduced from 128
LSTM_UNITS_1 = 128  # Reduced from 256
LSTM_UNITS_2 = 64  # Reduced from 128
EPOCHS = 20  # Reduced from 30
VALIDATION_SPLIT = 0.15
STRIDE = 5  # Increased from 3 for faster data prep

print("="*60)
print("OPTIMIZED CHARACTER-LEVEL BI-LSTM")
print("="*60)
print(f"\n🚀 Optimizations Applied:")
print(f"   Sequence Length: {SEQUENCE_LENGTH} (vs 100)")
print(f"   Embedding Dim: {EMBEDDING_DIM} (vs 128)")
print(f"   LSTM Units: {LSTM_UNITS_1}/{LSTM_UNITS_2} (vs 256/128)")
print(f"   Data Stride: {STRIDE} (vs 3)")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")

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

# Prepare sequences (faster with larger stride)
print(f"\n🔄 Creating training sequences (stride={STRIDE})...")
start_time = time.time()

X = []
y = []

for i in range(0, len(text) - SEQUENCE_LENGTH, STRIDE):
    sequence = text[i:i + SEQUENCE_LENGTH]
    target = text[i + SEQUENCE_LENGTH]
    X.append([char_to_idx[c] for c in sequence])
    y.append(char_to_idx[target])

X = np.array(X)
y = np.array(y)

data_prep_time = time.time() - start_time
print(f"Training sequences: {len(X):,}")
print(f"Data prep time: {data_prep_time:.1f}s")
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")

# Split data
split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"\n📊 Data split:")
print(f"  Training: {len(X_train):,} sequences")
print(f"  Validation: {len(X_val):,} sequences")

# Build optimized model
print("\n🏗️  Building optimized Bi-LSTM model...")
model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),
    Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True, dropout=0.2)),
    Bidirectional(LSTM(LSTM_UNITS_2, dropout=0.2)),
    Dropout(0.3),
    Dense(128, activation='relu'),  # Reduced from 256
    Dropout(0.3),
    Dense(vocab_size, activation='softmax')
], name='Optimized_CharLevel_BiLSTM')

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.002),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')]
)

print("\n📋 Model Summary:")
# Build model to show summary
model.build(input_shape=(None, SEQUENCE_LENGTH))
model.summary()

# Calculate model size
total_params = model.count_params()
model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
print(f"\n💾 Model Statistics:")
print(f"   Total parameters: {total_params:,}")
print(f"   Estimated size: {model_size_mb:.2f} MB")

# Callbacks
os.makedirs('models', exist_ok=True)
os.makedirs('logs/optimized', exist_ok=True)

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
        'models/optimized_bilstm_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
]

# Train
print("\n🚀 Starting training...")
print("="*60)

training_start = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - training_start

# Final evaluation
print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60)

train_loss, train_acc, train_top5 = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
val_loss, val_acc, val_top5 = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)

print(f"\n⏱️  Training Time: {training_time/3600:.2f} hours")

print(f"\n✅ Training Metrics:")
print(f"   Accuracy: {train_acc*100:.2f}%")
print(f"   Top-5 Accuracy: {train_top5*100:.2f}%")
print(f"   Loss: {train_loss:.4f}")

print(f"\n🎯 Validation Metrics:")
print(f"   Accuracy: {val_acc*100:.2f}%")
print(f"   Top-5 Accuracy: {val_top5*100:.2f}%")
print(f"   Loss: {val_loss:.4f}")
print(f"   Perplexity: {np.exp(val_loss):.2f}")

# Calculate if we reached target
target_reached = val_acc >= 0.70
print(f"\n🎯 Target Achievement (70% accuracy):")
print(f"   Status: {'✅ ACHIEVED' if target_reached else '🟡 IN PROGRESS'}")
print(f"   Progress: {val_acc/0.70*100:.1f}%")

# Generate sample text
print("\n📝 Sample text generation:")
print("-"*60)

def generate_text(model, seed_text, length=200, temperature=0.8):
    """Generate text using the trained model with temperature sampling"""
    generated = seed_text

    for _ in range(length):
        # Prepare input (use last SEQUENCE_LENGTH chars)
        input_seq = generated[-SEQUENCE_LENGTH:]
        # Pad if necessary
        if len(input_seq) < SEQUENCE_LENGTH:
            input_seq = ' ' * (SEQUENCE_LENGTH - len(input_seq)) + input_seq

        x = np.array([[char_to_idx.get(c, 0) for c in input_seq]])

        # Predict next character with temperature
        predictions = model.predict(x, verbose=0)[0]
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        next_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_idx]

        generated += next_char

    return generated

# Generate samples with different temperatures
for temp in [0.5, 0.8, 1.0]:
    print(f"\n🌡️  Temperature: {temp}")
    seed = "To be or not to be"
    generated_text = generate_text(model, seed, 150, temperature=temp)
    print(generated_text)
    print("-"*60)

print("\n✨ Training complete!")
print(f"Best model saved to: models/optimized_bilstm_best.h5")
print("="*60)

# Save char mappings
import pickle
with open('models/optimized_mappings.pkl', 'wb') as f:
    pickle.dump({
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'sequence_length': SEQUENCE_LENGTH,
        'vocab_size': vocab_size
    }, f)
print("Character mappings saved to: models/optimized_mappings.pkl")

# Save training summary
with open('logs/optimized/training_summary.txt', 'w') as f:
    f.write("OPTIMIZED BI-LSTM TRAINING SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  Sequence Length: {SEQUENCE_LENGTH}\n")
    f.write(f"  Embedding Dim: {EMBEDDING_DIM}\n")
    f.write(f"  LSTM Units: {LSTM_UNITS_1}/{LSTM_UNITS_2}\n")
    f.write(f"  Batch Size: {BATCH_SIZE}\n")
    f.write(f"  Epochs: {EPOCHS}\n")
    f.write(f"  Data Stride: {STRIDE}\n\n")
    f.write(f"Results:\n")
    f.write(f"  Training Time: {training_time/3600:.2f} hours\n")
    f.write(f"  Training Accuracy: {train_acc*100:.2f}%\n")
    f.write(f"  Validation Accuracy: {val_acc*100:.2f}%\n")
    f.write(f"  Validation Perplexity: {np.exp(val_loss):.2f}\n")
    f.write(f"  Target (70%) Reached: {target_reached}\n")

print("Training summary saved to: logs/optimized/training_summary.txt")
