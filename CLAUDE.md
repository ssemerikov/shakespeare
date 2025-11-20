# Shakespeare Text Generation: Optimized Bi-LSTM Implementation

## Project Overview

This project implements a high-performance Bidirectional LSTM (Bi-LSTM) model for text generation based on Shakespeare's complete works. The implementation focuses on two key objectives:

1. **Maximum Accuracy**: Achieving the best possible text generation quality
2. **Optimal Inference Speed**: Ensuring fast prediction for real-time applications

## Architecture Evolution

### Current Implementation (Baseline)
- **Model Type**: SimpleRNN
- **Parameters**: ~20M
- **Embedding Dim**: 256
- **Hidden Units**: 512
- **Sequence Length**: 20
- **Validation Accuracy**: ~7.7% (epoch 1) → ~5.1% (epoch 25, overfitting)

### Proposed Bi-LSTM Implementation

```
Input (batch_size, seq_len)
    ↓
Embedding Layer (vocab_size=25751, emb_dim=256)
    ↓
Spatial Dropout (0.2)
    ↓
Bi-LSTM Layer 1 (256 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
Bi-LSTM Layer 2 (128 units, return_sequences=False)
    ↓
Dropout (0.3)
    ↓
Dense Layer (512 units, ReLU)
    ↓
Dropout (0.4)
    ↓
Output Dense (vocab_size, Softmax)
```

## Key Optimizations

### 1. Accuracy Improvements

#### A. Model Architecture
- **Bidirectional LSTM**: Captures context from both past and future
- **Stacked Layers**: Two Bi-LSTM layers for deeper representation
- **Regularization**: Dropout layers (0.2-0.4) to prevent overfitting
- **Dense Hidden Layer**: Additional capacity for complex patterns

#### B. Training Strategy
```python
# Learning rate scheduling
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Gradient clipping
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
```

#### C. Data Preprocessing
- **Improved Tokenization**: Better handling of punctuation and special characters
- **Dynamic Sequence Length**: Variable length sequences (10-50 tokens)
- **Data Augmentation**: Text perturbation and back-translation
- **Vocabulary Optimization**: Focus on top 10K words with subword tokenization

#### D. Advanced Techniques
- **Label Smoothing**: Reduces overconfidence (epsilon=0.1)
- **Gradient Accumulation**: Simulate larger batch sizes
- **Mixed Precision Training**: Faster training with FP16

### 2. Inference Speed Optimizations

#### A. Model Compression
```python
# Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
# Expected: 4x size reduction, 2-3x speed improvement
```

#### B. Architecture Modifications
- **GRU Alternative**: 25% faster than LSTM, similar accuracy
- **Reduced Hidden Dimensions**: 256→128→64 cascade
- **Pruning**: Remove 30-40% of weights with minimal accuracy loss

#### C. Runtime Optimizations
```python
# Batch inference
def batch_predict(model, texts, batch_size=32):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        preds = model.predict(batch, verbose=0)
        predictions.extend(preds)
    return predictions

# Compiled prediction function
@tf.function(input_signature=[tf.TensorSpec(shape=[None, SEQUENCE_LENGTH], dtype=tf.int32)])
def fast_predict(inputs):
    return model(inputs, training=False)
```

#### D. Deployment Strategies
- **ONNX Runtime**: 2-5x faster inference
- **TensorRT**: GPU optimization for production
- **Model Serving**: TensorFlow Serving with gRPC
- **Edge Deployment**: TFLite for mobile/embedded

## Implementation Plan

### Phase 1: Enhanced Data Pipeline (Week 1)
```python
class ShakespeareDataset:
    """Optimized data pipeline with caching and prefetching"""

    def __init__(self, file_path, vocab_size=10000):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size,
            oov_token='<UNK>',
            filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
        )
        self.load_and_preprocess(file_path)

    def create_tf_dataset(self, sequence_length, batch_size=64):
        """Create optimized tf.data pipeline"""
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
```

### Phase 2: Bi-LSTM Model Implementation (Week 1-2)
```python
def build_bilstm_model(vocab_size, embedding_dim=256, seq_length=20):
    """Build optimized Bi-LSTM model"""

    model = Sequential([
        # Embedding layer with mask_zero for variable length
        Embedding(vocab_size, embedding_dim, mask_zero=True),
        SpatialDropout1D(0.2),

        # First Bi-LSTM layer
        Bidirectional(LSTM(256, return_sequences=True,
                          dropout=0.2, recurrent_dropout=0.2)),

        # Second Bi-LSTM layer
        Bidirectional(LSTM(128, return_sequences=False,
                          dropout=0.2, recurrent_dropout=0.2)),

        # Dense layers
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.4),
        Dense(vocab_size, activation='softmax')
    ])

    return model
```

### Phase 3: Training with Optimizations (Week 2)
```python
def train_optimized_model(model, train_dataset, val_dataset, epochs=50):
    """Train with advanced optimizations"""

    # Mixed precision training
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),
        TensorBoard(log_dir='./logs'),
        CSVLogger('training_history.csv')
    ]

    # Compile with label smoothing
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return model, history
```

### Phase 4: Speed Optimization (Week 3)
```python
# Model quantization
def quantize_model(model, representative_dataset):
    """Apply post-training quantization"""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for calibration
    def representative_data_gen():
        for input_value in representative_dataset.take(100):
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    quantized_model = converter.convert()
    return quantized_model

# Pruning
def prune_model(model, train_dataset):
    """Apply magnitude-based pruning"""

    import tensorflow_model_optimization as tfmot

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000
        )
    }

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model, **pruning_params
    )

    model_for_pruning.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model_for_pruning
```

### Phase 5: Inference Pipeline (Week 3-4)
```python
class OptimizedTextGenerator:
    """High-performance text generation"""

    def __init__(self, model_path, tokenizer, use_tflite=False):
        self.tokenizer = tokenizer
        self.use_tflite = use_tflite

        if use_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        else:
            self.model = tf.keras.models.load_model(model_path)
            # Convert to TF function for graph optimization
            self.predict_fn = tf.function(self.model,
                input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])

    @tf.function
    def generate_next_word_fast(self, sequence):
        """Optimized single word prediction"""
        if self.use_tflite:
            return self._predict_tflite(sequence)
        else:
            logits = self.predict_fn(sequence)
            return tf.argmax(logits, axis=-1)

    def generate_text(self, prompt, num_words=50, temperature=1.0, top_k=40):
        """Generate text with sampling strategies"""

        tokens = self.tokenizer.texts_to_sequences([prompt])[0]
        generated = tokens.copy()

        for _ in range(num_words):
            # Prepare input
            input_seq = np.array([generated[-self.seq_length:]])

            # Predict
            predictions = self.model.predict(input_seq, verbose=0)[0]

            # Apply temperature
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

            # Top-k sampling
            top_k_indices = np.argpartition(predictions, -top_k)[-top_k:]
            top_k_probs = predictions[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs)

            # Sample
            next_token = np.random.choice(top_k_indices, p=top_k_probs)
            generated.append(next_token)

        return self.tokenizer.sequences_to_texts([generated])[0]
```

## Performance Benchmarks

### Expected Metrics

| Metric | SimpleRNN (Baseline) | Bi-LSTM (Target) | Bi-LSTM + Optimizations |
|--------|---------------------|------------------|------------------------|
| **Accuracy** |
| Training Accuracy | ~5.6% | ~12-15% | ~15-18% |
| Validation Accuracy | ~5.1% | ~10-12% | ~12-15% |
| Perplexity | ~450 | ~200-250 | ~180-220 |
| **Speed** |
| Training Time/Epoch | 135s | 180-200s | 160-180s |
| Inference (single) | ~50ms | ~80ms | ~30ms (optimized) |
| Inference (batch 32) | - | ~150ms | ~60ms (optimized) |
| **Model Size** |
| Float32 | 77 MB | 85-95 MB | 22 MB (quantized) |
| Quantized INT8 | - | - | 22 MB |
| **Generated Text Quality** |
| Coherence (1-10) | 3 | 6-7 | 7-8 |
| Grammar Score | 40% | 65-70% | 70-75% |

### Benchmark Tests
```python
def benchmark_model(model, test_sequences, batch_sizes=[1, 8, 32, 64]):
    """Comprehensive benchmarking"""

    results = {}

    for batch_size in batch_sizes:
        # Prepare batch
        batch = test_sequences[:batch_size]

        # Warmup
        for _ in range(10):
            _ = model.predict(batch, verbose=0)

        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            _ = model.predict(batch, verbose=0)
            times.append(time.time() - start)

        results[batch_size] = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'throughput': batch_size / np.mean(times)
        }

    return results
```

## Advanced Features

### 1. Attention Mechanism
```python
class AttentionLayer(Layer):
    """Bahdanau attention for improved context"""

    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        # query: (batch, hidden)
        # values: (batch, seq_len, hidden)

        query_with_time = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * values
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights
```

### 2. Beam Search Decoding
```python
def beam_search(model, start_sequence, beam_width=5, max_length=50):
    """Beam search for better generation quality"""

    sequences = [[start_sequence, 0.0]]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            if len(seq) >= max_length:
                all_candidates.append([seq, score])
                continue

            # Get predictions
            input_seq = np.array([seq[-SEQUENCE_LENGTH:]])
            predictions = model.predict(input_seq, verbose=0)[0]

            # Get top-k
            top_k_indices = np.argpartition(predictions, -beam_width)[-beam_width:]

            for idx in top_k_indices:
                candidate = [seq + [idx], score + np.log(predictions[idx] + 1e-10)]
                all_candidates.append(candidate)

        # Select top sequences
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]

    return sequences[0][0]
```

### 3. Transfer Learning
```python
def load_pretrained_embeddings(embedding_matrix, word_index, glove_path):
    """Load GloVe or Word2Vec embeddings"""

    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Fill embedding matrix
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
```

## File Structure

```
shakespeare/
├── CLAUDE.md                    # This file
├── Shakespeare.ipynb            # Original notebook
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Data loading and preprocessing
│   │   └── dataset.py          # TF Dataset creation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bilstm.py          # Bi-LSTM model definition
│   │   ├── attention.py       # Attention layers
│   │   └── losses.py          # Custom loss functions
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py           # Training loop
│   │   └── callbacks.py       # Custom callbacks
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generator.py       # Text generation
│   │   └── optimize.py        # Model optimization
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py         # Custom metrics
│       └── visualization.py   # Training visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_optimization.ipynb
├── configs/
│   ├── base_config.yaml
│   ├── bilstm_config.yaml
│   └── optimization_config.yaml
├── scripts/
│   ├── train_model.py
│   ├── optimize_model.py
│   ├── benchmark.py
│   └── export_model.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_inference.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation & Usage

### Installation
```bash
# Clone repository
git clone <repo-url>
cd shakespeare

# Install dependencies
pip install -r requirements.txt

# For optimization tools
pip install tensorflow-model-optimization
pip install onnx onnxruntime
```

### Quick Start
```python
# Train model
from src.training.train import train_bilstm_model

model, history = train_bilstm_model(
    data_path='Shakespeare.txt',
    config='configs/bilstm_config.yaml',
    epochs=50
)

# Generate text
from src.inference.generator import OptimizedTextGenerator

generator = OptimizedTextGenerator(model_path='models/best_model.h5')
text = generator.generate_text(
    prompt="To be or not to be",
    num_words=100,
    temperature=0.8
)
print(text)

# Optimize for deployment
from src.inference.optimize import optimize_for_deployment

optimize_for_deployment(
    model_path='models/best_model.h5',
    output_dir='models/optimized/',
    quantize=True,
    prune=True
)
```

### Command Line Interface
```bash
# Training
python scripts/train_model.py \
    --data Shakespeare.txt \
    --config configs/bilstm_config.yaml \
    --epochs 50 \
    --batch-size 64

# Optimization
python scripts/optimize_model.py \
    --model models/best_model.h5 \
    --output models/optimized/ \
    --quantize --prune

# Benchmarking
python scripts/benchmark.py \
    --model models/best_model.h5 \
    --optimized models/optimized/quantized.tflite

# Text generation
python scripts/generate.py \
    --model models/best_model.h5 \
    --prompt "To be or not to be" \
    --length 100 \
    --temperature 0.8
```

## Configuration

### Example Config (bilstm_config.yaml)
```yaml
model:
  architecture: bilstm
  embedding_dim: 256
  lstm_units: [256, 128]
  dense_units: 512
  dropout_rates: [0.2, 0.3, 0.4]
  bidirectional: true

data:
  vocab_size: 10000
  sequence_length: 20
  batch_size: 64
  train_split: 0.8

training:
  epochs: 50
  initial_lr: 0.001
  optimizer: adam
  loss: sparse_categorical_crossentropy
  metrics: [accuracy, perplexity]

callbacks:
  early_stopping:
    patience: 5
    monitor: val_loss
  lr_scheduler:
    factor: 0.5
    patience: 2
    min_lr: 1.0e-6
  model_checkpoint:
    save_best_only: true
    monitor: val_accuracy

optimization:
  mixed_precision: true
  gradient_clipping: 1.0
  label_smoothing: 0.1

inference:
  temperature: 0.8
  top_k: 40
  beam_width: 5
```

## Evaluation Metrics

### Text Quality Metrics
```python
def evaluate_text_quality(generated_texts, reference_texts):
    """Comprehensive quality evaluation"""

    metrics = {
        'bleu': calculate_bleu_scores(generated_texts, reference_texts),
        'perplexity': calculate_perplexity(model, test_data),
        'diversity': calculate_diversity(generated_texts),
        'coherence': calculate_coherence_score(generated_texts),
        'grammar': calculate_grammar_score(generated_texts)
    }

    return metrics

def calculate_diversity(texts, n=4):
    """Calculate n-gram diversity"""
    all_ngrams = []
    for text in texts:
        words = text.split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        all_ngrams.extend(ngrams)

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)

    return unique_ngrams / total_ngrams if total_ngrams > 0 else 0
```

## Future Enhancements

### Short-term (1-2 months)
- [ ] Implement Transformer-based architecture
- [ ] Add GPT-style decoder-only model
- [ ] Integrate pre-trained embeddings (GloVe, FastText)
- [ ] Multi-GPU training support
- [ ] Hyperparameter tuning with Optuna

### Medium-term (3-6 months)
- [ ] Fine-tuning on different literary styles
- [ ] Multi-task learning (generation + classification)
- [ ] Conditional generation (style, genre, sentiment)
- [ ] API service with FastAPI
- [ ] Web interface for interactive generation

### Long-term (6-12 months)
- [ ] GPT-3 style architecture
- [ ] Reinforcement learning from human feedback (RLHF)
- [ ] Few-shot learning capabilities
- [ ] Multi-modal generation (text + image descriptions)
- [ ] Distributed training on cloud infrastructure

## References

### Academic Papers
1. **Bi-LSTM**: "Bidirectional LSTM Networks for Improved Phoneme Classification and Recognition" - Graves & Schmidhuber (2005)
2. **Attention**: "Neural Machine Translation by Jointly Learning to Align and Translate" - Bahdanau et al. (2014)
3. **Optimization**: "Mixed Precision Training" - Micikevicius et al. (2017)
4. **Quantization**: "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" - Jacob et al. (2018)

### Libraries & Tools
- TensorFlow 2.x: https://www.tensorflow.org/
- TensorFlow Model Optimization: https://www.tensorflow.org/model_optimization
- ONNX Runtime: https://onnxruntime.ai/
- TensorRT: https://developer.nvidia.com/tensorrt

### Datasets
- Project Gutenberg Shakespeare: https://www.gutenberg.org/ebooks/100

## License
MIT License - See LICENSE file for details

## Contributors
- Primary Developer: AI Assistant (Claude)
- Based on original notebook by: [Original Author]

## Changelog

### v2.0.0 (Planned) - Bi-LSTM Implementation
- Replace SimpleRNN with Bi-LSTM architecture
- Add attention mechanism
- Implement advanced optimization techniques
- Improve inference speed by 3-5x
- Increase accuracy by 2-3x

### v1.0.0 (Current) - SimpleRNN Baseline
- Basic text generation with SimpleRNN
- ~20M parameters
- ~5% validation accuracy

---

**Last Updated**: 2025-11-20
**Status**: Planning & Design Phase
**Next Milestone**: Phase 1 - Enhanced Data Pipeline
