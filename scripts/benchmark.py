#!/usr/bin/env python3
"""
Benchmark script for model evaluation.
"""

import os
import sys
import argparse
import pickle
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from src.data.preprocessing import ShakespeareDataPreprocessor
from src.data.dataset import create_tf_dataset
from src.utils.metrics import benchmark_inference, evaluate_model, print_model_summary


def main(model_path, tokenizer_path, data_file):
    """
    Benchmark model performance.

    Args:
        model_path: Path to trained model
        tokenizer_path: Path to tokenizer
        data_file: Path to data file for testing
    """
    print("="*60)
    print("MODEL BENCHMARK")
    print("="*60)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Print model summary
    print_model_summary(model)

    # Prepare test data
    print("\n" + "="*60)
    print("PREPARING TEST DATA")
    print("="*60)

    preprocessor = ShakespeareDataPreprocessor(
        vocab_size=len(tokenizer.word_index) + 1,
        sequence_length=model.layers[0].input_shape[1],
        oov_token='<UNK>'
    )
    preprocessor.tokenizer = tokenizer

    X_train, y_train, X_val, y_val = preprocessor.prepare_data(
        file_path=data_file,
        train_split=0.8
    )

    # Create validation dataset
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=64, shuffle=False)

    # Evaluate model
    metrics = evaluate_model(model, val_dataset)

    # Benchmark inference speed
    print("\nPreparing test sequences for speed benchmark...")
    test_sequences = X_val[:1000]  # Use first 1000 sequences

    results = benchmark_inference(
        model=model,
        test_sequences=test_sequences,
        batch_sizes=[1, 8, 32, 64],
        num_runs=100
    )

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"\nAccuracy Metrics:")
    print(f"  Validation Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Top-5 Accuracy: {metrics.get('top_5_accuracy', 0):.4f}")
    print(f"  Perplexity: {metrics.get('perplexity', 0):.2f}")
    print(f"\nSpeed Metrics (batch_size=1):")
    if 1 in results:
        print(f"  Mean inference time: {results[1]['mean_ms']:.2f} ms")
        print(f"  Throughput: {results[1]['throughput']:.2f} samples/sec")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark model performance')
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_bilstm_model.h5',
        help='Path to trained model'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='models/best_bilstm_model_tokenizer.pkl',
        help='Path to tokenizer'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='Shakespeare.txt',
        help='Path to data file'
    )

    args = parser.parse_args()
    main(args.model, args.tokenizer, args.data)
