"""
Evaluation metrics and utilities.
"""

import time
import numpy as np


def calculate_perplexity(loss):
    """
    Calculate perplexity from loss.

    Args:
        loss: Cross-entropy loss

    Returns:
        Perplexity value
    """
    return np.exp(loss)


def benchmark_inference(model, test_sequences, batch_sizes=[1, 8, 32, 64], num_runs=100):
    """
    Benchmark model inference speed.

    Args:
        model: Trained model
        test_sequences: Test sequences for benchmarking
        batch_sizes: List of batch sizes to test
        num_runs: Number of runs for averaging

    Returns:
        Dictionary of benchmark results
    """
    results = {}

    print("\n" + "="*60)
    print("INFERENCE SPEED BENCHMARK")
    print("="*60)

    for batch_size in batch_sizes:
        if len(test_sequences) < batch_size:
            continue

        # Prepare batch
        batch = test_sequences[:batch_size]

        # Warmup
        print(f"\nBatch size {batch_size}: Warming up...")
        for _ in range(10):
            _ = model.predict(batch, verbose=0)

        # Benchmark
        print(f"Running {num_runs} iterations...")
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = model.predict(batch, verbose=0)
            end = time.time()
            times.append(end - start)

        mean_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        throughput = batch_size / np.mean(times)

        results[batch_size] = {
            'mean_ms': mean_time,
            'std_ms': std_time,
            'throughput': throughput,
            'per_sample_ms': mean_time / batch_size
        }

        print(f"  Mean time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"  Per sample: {mean_time/batch_size:.2f} ms")
        print(f"  Throughput: {throughput:.2f} samples/sec")

    print("="*60)
    return results


def evaluate_model(model, val_dataset):
    """
    Evaluate model on validation set.

    Args:
        model: Trained model
        val_dataset: Validation dataset

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    results = model.evaluate(val_dataset, verbose=1)
    metrics = dict(zip(model.metrics_names, results))

    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    if 'loss' in metrics:
        perplexity = calculate_perplexity(metrics['loss'])
        metrics['perplexity'] = perplexity
        print(f"  perplexity: {perplexity:.2f}")

    print("="*60)
    return metrics


def print_model_summary(model):
    """
    Print detailed model summary.

    Args:
        model: Keras model
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    try:
        print(f"\nTotal parameters: {model.count_params():,}")
    except ValueError:
        print("\nModel not yet built - parameters will be shown after first batch")
    print("="*60)
