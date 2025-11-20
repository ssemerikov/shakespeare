"""
Training module with optimization callbacks.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
)


def create_callbacks(
    model_save_path='models/best_model.h5',
    log_dir='logs',
    patience_lr=2,
    patience_early=5,
    factor_lr=0.5,
    min_lr=1e-6
):
    """
    Create training callbacks for optimization.

    Args:
        model_save_path: Path to save best model
        log_dir: Directory for TensorBoard logs
        patience_lr: Patience for learning rate reduction
        patience_early: Patience for early stopping
        factor_lr: Factor to reduce learning rate
        min_lr: Minimum learning rate

    Returns:
        List of callbacks
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor_lr,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
            mode='min'
        ),

        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=patience_early,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),

        # Save best model
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),

        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        ),

        # CSV logging
        CSVLogger(
            os.path.join(log_dir, 'training_history.csv'),
            append=True
        )
    ]

    return callbacks


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs=50,
    callbacks=None,
    verbose=1
):
    """
    Train the model with given datasets.

    Args:
        model: Compiled Keras model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        callbacks: List of callbacks
        verbose: Verbosity level

    Returns:
        Training history
    """
    print(f"\nStarting training for {epochs} epochs...")
    try:
        print(f"Model parameters: {model.count_params():,}")
    except ValueError:
        print("Model parameters will be shown after first batch")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )

    print("\nTraining complete!")

    return history


def train_bilstm_model(
    model,
    train_dataset,
    val_dataset,
    epochs=50,
    model_save_path='models/best_bilstm_model.h5',
    log_dir='logs'
):
    """
    Complete training pipeline for Bi-LSTM model.

    Args:
        model: Compiled Bi-LSTM model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of epochs
        model_save_path: Path to save model
        log_dir: Log directory

    Returns:
        Tuple of (model, history)
    """
    # Create callbacks
    callbacks = create_callbacks(
        model_save_path=model_save_path,
        log_dir=log_dir
    )

    # Train model
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    return model, history
