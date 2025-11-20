"""
Bidirectional LSTM model for text generation.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.regularizers import l2


def build_bilstm_model(
    vocab_size,
    embedding_dim=256,
    sequence_length=20,
    lstm_units=[256, 128],
    dense_units=512,
    dropout_rates=[0.2, 0.3, 0.4],
    embedding_matrix=None,
    trainable_embeddings=True
):
    """
    Build Bidirectional LSTM model for text generation.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        sequence_length: Length of input sequences
        lstm_units: List of LSTM units for each layer
        dense_units: Number of units in dense layer
        dropout_rates: List of dropout rates [spatial, lstm, dense]
        embedding_matrix: Pre-trained embedding matrix (optional)
        trainable_embeddings: Whether to train embeddings

    Returns:
        Compiled Keras model
    """
    model = Sequential(name='BiLSTM_TextGen')

    # Embedding layer
    if embedding_matrix is not None:
        print(f"Using pre-trained embeddings (trainable={trainable_embeddings})")
        model.add(Embedding(
            vocab_size,
            embedding_dim,
            weights=[embedding_matrix],
            trainable=trainable_embeddings,
            mask_zero=True,
            name='embedding'
        ))
    else:
        print("Training embeddings from scratch")
        model.add(Embedding(
            vocab_size,
            embedding_dim,
            mask_zero=True,
            name='embedding'
        ))

    # Spatial dropout
    model.add(SpatialDropout1D(dropout_rates[0], name='spatial_dropout'))

    # First Bi-LSTM layer
    model.add(Bidirectional(
        LSTM(
            lstm_units[0],
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='lstm_1'
        ),
        name='bidirectional_1'
    ))

    # Dropout
    model.add(Dropout(dropout_rates[1], name='dropout_1'))

    # Second Bi-LSTM layer
    if len(lstm_units) > 1:
        model.add(Bidirectional(
            LSTM(
                lstm_units[1],
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm_2'
            ),
            name='bidirectional_2'
        ))
        model.add(Dropout(dropout_rates[1], name='dropout_2'))

    # Dense layer with L2 regularization
    model.add(Dense(
        dense_units,
        activation='relu',
        kernel_regularizer=l2(1e-4),
        name='dense_hidden'
    ))

    # Final dropout
    model.add(Dropout(dropout_rates[2], name='dropout_final'))

    # Output layer
    model.add(Dense(vocab_size, activation='softmax', name='output'))

    return model


def compile_model(model, learning_rate=0.001, clipnorm=1.0, label_smoothing=0.1):
    """
    Compile the model with optimizer and loss function.

    Args:
        model: Keras model to compile
        learning_rate: Initial learning rate
        clipnorm: Gradient clipping norm
        label_smoothing: Label smoothing factor

    Returns:
        Compiled model
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )

    return model


def create_bilstm_model(
    vocab_size,
    embedding_dim=256,
    sequence_length=20,
    lstm_units=[256, 128],
    dense_units=512,
    dropout_rates=[0.2, 0.3, 0.4],
    embedding_matrix=None,
    trainable_embeddings=True,
    learning_rate=0.001
):
    """
    Build and compile Bi-LSTM model.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        sequence_length: Length of input sequences
        lstm_units: List of LSTM units for each layer
        dense_units: Number of units in dense layer
        dropout_rates: List of dropout rates
        embedding_matrix: Pre-trained embedding matrix
        trainable_embeddings: Whether to train embeddings
        learning_rate: Initial learning rate

    Returns:
        Compiled Keras model
    """
    model = build_bilstm_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        sequence_length=sequence_length,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rates=dropout_rates,
        embedding_matrix=embedding_matrix,
        trainable_embeddings=trainable_embeddings
    )

    model = compile_model(model, learning_rate=learning_rate)

    return model
