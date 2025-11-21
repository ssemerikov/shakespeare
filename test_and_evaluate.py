#!/usr/bin/env python3
"""
Comprehensive testing and evaluation for character-level Bi-LSTM model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import time
from collections import Counter
import re

print("="*70)
print("SHAKESPEARE BI-LSTM: COMPREHENSIVE MODEL EVALUATION")
print("="*70)

# Load model and mappings
print("\n📦 Loading model and character mappings...")
try:
    model = keras.models.load_model('models/char_bilstm_best.h5')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Try to load mappings, or create them if they don't exist
try:
    with open('models/char_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
        char_to_idx = mappings['char_to_idx']
        idx_to_char = mappings['idx_to_char']
    print(f"✓ Character mappings loaded (vocab size: {len(char_to_idx)})")
except FileNotFoundError:
    print("⚠ Character mappings not found, creating from Shakespeare.txt...")
    with open('Shakespeare.txt', 'r', encoding='utf-8') as f:
        text_for_mappings = f.read()
    chars = sorted(list(set(text_for_mappings)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    # Save for future use
    with open('models/char_mappings.pkl', 'wb') as f:
        pickle.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)
    print(f"✓ Character mappings created (vocab size: {len(char_to_idx)})")

# Load test data
print("\n📖 Loading Shakespeare text for testing...")
with open('Shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Use last 15% as test set
test_start = int(len(text) * 0.85)
test_text = text[test_start:]
print(f"✓ Test set size: {len(test_text):,} characters")

# ============================================================================
# 1. QUANTITATIVE METRICS
# ============================================================================
print("\n" + "="*70)
print("1. QUANTITATIVE METRICS")
print("="*70)

# Prepare test sequences
SEQUENCE_LENGTH = 100
test_sequences = []
test_targets = []

for i in range(0, min(10000, len(test_text) - SEQUENCE_LENGTH), 10):
    sequence = test_text[i:i + SEQUENCE_LENGTH]
    target = test_text[i + SEQUENCE_LENGTH]

    try:
        test_sequences.append([char_to_idx.get(c, 0) for c in sequence])
        test_targets.append(char_to_idx.get(target, 0))
    except:
        continue

X_test = np.array(test_sequences)
y_test = np.array(test_targets)

print(f"\nTest sequences: {len(X_test):,}")
print(f"Evaluating model performance...")

# Evaluate
start_time = time.time()
test_loss, test_acc, test_top5 = model.evaluate(X_test, y_test, batch_size=512, verbose=0)
eval_time = time.time() - start_time

print(f"\n📊 Test Set Performance:")
print(f"  ├─ Accuracy: {test_acc*100:.2f}%")
print(f"  ├─ Top-5 Accuracy: {test_top5*100:.2f}%")
print(f"  ├─ Loss: {test_loss:.4f}")
print(f"  └─ Evaluation Time: {eval_time:.2f}s")

# Perplexity
perplexity = np.exp(test_loss)
print(f"\n📈 Perplexity: {perplexity:.2f}")
print(f"   (Lower is better. Good models: 10-50, Excellent: <10)")

# ============================================================================
# 2. INFERENCE SPEED BENCHMARKING
# ============================================================================
print("\n" + "="*70)
print("2. INFERENCE SPEED BENCHMARKING")
print("="*70)

# Single prediction
sample_input = X_test[:1]
warmup_runs = 10
for _ in range(warmup_runs):
    _ = model.predict(sample_input, verbose=0)

single_times = []
for _ in range(100):
    start = time.time()
    _ = model.predict(sample_input, verbose=0)
    single_times.append(time.time() - start)

print(f"\n⚡ Single Character Prediction:")
print(f"  ├─ Mean: {np.mean(single_times)*1000:.2f}ms")
print(f"  ├─ Std: {np.std(single_times)*1000:.2f}ms")
print(f"  └─ Throughput: {1/np.mean(single_times):.1f} predictions/sec")

# Batch predictions
for batch_size in [32, 64, 128, 256]:
    batch_input = X_test[:batch_size]

    batch_times = []
    for _ in range(20):
        start = time.time()
        _ = model.predict(batch_input, verbose=0)
        batch_times.append(time.time() - start)

    mean_time = np.mean(batch_times)
    throughput = batch_size / mean_time

    print(f"\n📦 Batch Size {batch_size}:")
    print(f"  ├─ Mean Time: {mean_time*1000:.2f}ms")
    print(f"  └─ Throughput: {throughput:.1f} predictions/sec")

# ============================================================================
# 3. TEXT GENERATION QUALITY
# ============================================================================
print("\n" + "="*70)
print("3. TEXT GENERATION QUALITY")
print("="*70)

def generate_text(model, seed_text, length=200, temperature=1.0):
    """Generate text with temperature sampling"""
    generated = seed_text

    for _ in range(length):
        # Prepare input
        x = np.array([[char_to_idx.get(c, 0) for c in generated[-SEQUENCE_LENGTH:]]])

        # Predict
        predictions = model.predict(x, verbose=0)[0]

        # Apply temperature
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        # Sample
        next_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_idx]

        generated += next_char

    return generated

# Test different temperatures
test_prompts = [
    "To be or not to be",
    "O Romeo, Romeo",
    "Friends, Romans, countrymen",
    "All the world's a stage"
]

for temp in [0.5, 0.8, 1.0, 1.2]:
    print(f"\n{'─'*70}")
    print(f"Temperature: {temp}")
    print(f"{'─'*70}")

    for prompt in test_prompts:
        print(f"\n🎭 Prompt: '{prompt}'")
        generated = generate_text(model, prompt, length=150, temperature=temp)
        print(f"Generated:\n{generated}")
        print()

# ============================================================================
# 4. QUALITATIVE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("4. QUALITATIVE ANALYSIS")
print("="*70)

# Generate longer sample
print("\n📝 Extended Sample (500 characters, temp=0.8):")
print("─"*70)
long_sample = generate_text(model, "To be or not to be", length=500, temperature=0.8)
print(long_sample)
print("─"*70)

# Analyze generated text
def analyze_text(text):
    """Analyze text quality metrics"""
    # Word count
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    unique_words = len(set(words))

    # Character distribution
    char_dist = Counter(text)

    # Sentence-like structures (rough estimate)
    sentences = text.count('.') + text.count('!') + text.count('?')

    # Average word length
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    return {
        'word_count': word_count,
        'unique_words': unique_words,
        'vocabulary_diversity': unique_words / word_count if word_count > 0 else 0,
        'char_distribution': char_dist,
        'sentences': sentences,
        'avg_word_length': avg_word_len
    }

print("\n📊 Generated Text Analysis:")
analysis = analyze_text(long_sample)
print(f"  ├─ Total Words: {analysis['word_count']}")
print(f"  ├─ Unique Words: {analysis['unique_words']}")
print(f"  ├─ Vocabulary Diversity: {analysis['vocabulary_diversity']:.2%}")
print(f"  ├─ Average Word Length: {analysis['avg_word_length']:.1f} characters")
print(f"  └─ Sentence-like Structures: {analysis['sentences']}")

# Compare with Shakespeare
print("\n📚 Shakespeare Original (for comparison):")
original_analysis = analyze_text(test_text[:len(long_sample)])
print(f"  ├─ Total Words: {original_analysis['word_count']}")
print(f"  ├─ Unique Words: {original_analysis['unique_words']}")
print(f"  ├─ Vocabulary Diversity: {original_analysis['vocabulary_diversity']:.2%}")
print(f"  ├─ Average Word Length: {original_analysis['avg_word_length']:.1f} characters")
print(f"  └─ Sentence-like Structures: {original_analysis['sentences']}")

# ============================================================================
# 5. MODEL ARCHITECTURE SUMMARY
# ============================================================================
print("\n" + "="*70)
print("5. MODEL ARCHITECTURE SUMMARY")
print("="*70)

print("\n🏗️ Model Architecture:")
model.summary()

# Parameter count
total_params = model.count_params()
model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

print(f"\n📦 Model Statistics:")
print(f"  ├─ Total Parameters: {total_params:,}")
print(f"  ├─ Model Size (FP32): {model_size_mb:.2f} MB")
print(f"  ├─ Vocabulary Size: {len(char_to_idx)}")
print(f"  └─ Sequence Length: {SEQUENCE_LENGTH}")

# ============================================================================
# 6. FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("6. FINAL SUMMARY")
print("="*70)

print(f"""
✅ MODEL EVALUATION COMPLETE

Key Metrics:
  • Test Accuracy: {test_acc*100:.2f}%
  • Top-5 Accuracy: {test_top5*100:.2f}%
  • Perplexity: {perplexity:.2f}
  • Inference Speed: {1/np.mean(single_times):.1f} predictions/sec
  • Model Size: {model_size_mb:.2f} MB

Quality Assessment:
  • Vocabulary Diversity: {analysis['vocabulary_diversity']:.1%} (Original: {original_analysis['vocabulary_diversity']:.1%})
  • Average Word Length: {analysis['avg_word_length']:.1f} chars (Original: {original_analysis['avg_word_length']:.1f})

Performance Level:
""")

if test_acc >= 0.70:
    print("  🌟 EXCELLENT - Exceeds 70% accuracy target!")
elif test_acc >= 0.60:
    print("  ✅ GOOD - Strong performance, near target")
elif test_acc >= 0.50:
    print("  ⚠️  MODERATE - Decent but below target")
else:
    print("  ❌ NEEDS IMPROVEMENT - More training required")

print("\n" + "="*70)
print("Report saved to: evaluation_report.txt")
print("="*70)
