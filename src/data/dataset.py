"""
TensorFlow Dataset creation module for optimized data pipeline.
"""

import tensorflow as tf


def create_tf_dataset(X, y, batch_size=64, shuffle=True, buffer_size=10000):
    """
    Create optimized TensorFlow dataset.

    Args:
        X: Input sequences
        y: Target tokens
        batch_size: Batch size
        shuffle: Whether to shuffle data
        buffer_size: Shuffle buffer size

    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_train_val_datasets(X_train, y_train, X_val, y_val, batch_size=64):
    """
    Create training and validation datasets.

    Args:
        X_train: Training input sequences
        y_train: Training target tokens
        X_val: Validation input sequences
        y_val: Validation target tokens
        batch_size: Batch size

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset
