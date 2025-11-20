#!/usr/bin/env python3
"""
Main training script for Bi-LSTM text generation model.
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import ShakespeareDataPreprocessor
from src.data.dataset import create_train_val_datasets
from src.models.bilstm import create_bilstm_model
from src.training.train import train_bilstm_model
from src.utils.embeddings import prepare_pretrained_embeddings
from src.utils.config import load_config, get_model_config, get_data_config, get_training_config, get_paths_config
from src.utils.metrics import print_model_summary, evaluate_model


def main(config_path):
    """
    Main training pipeline.

    Args:
        config_path: Path to configuration file
    """
    print("="*60)
    print("BI-LSTM SHAKESPEARE TEXT GENERATION")
    print("="*60)

    # Load configuration
    print(f"\nLoading configuration from {config_path}...")
    config = load_config(config_path)

    model_config = get_model_config(config)
    data_config = get_data_config(config)
    training_config = get_training_config(config)
    paths_config = get_paths_config(config)

    # Data preprocessing
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)

    preprocessor = ShakespeareDataPreprocessor(
        vocab_size=data_config['vocab_size'],
        sequence_length=data_config['sequence_length'],
        oov_token='<UNK>'
    )

    X_train, y_train, X_val, y_val = preprocessor.prepare_data(
        file_path=data_config['text_file'],
        train_split=data_config['train_split']
    )

    # Create TF datasets
    print("\nCreating TensorFlow datasets...")
    train_dataset, val_dataset = create_train_val_datasets(
        X_train, y_train, X_val, y_val,
        batch_size=data_config['batch_size']
    )

    # Prepare embeddings
    embedding_matrix = None
    if model_config.get('use_pretrained_embeddings', False):
        print("\n" + "="*60)
        print("PREPARING TRANSFER LEARNING EMBEDDINGS")
        print("="*60)

        embedding_matrix = prepare_pretrained_embeddings(
            word_index=preprocessor.get_word_index(),
            vocab_size=preprocessor.vocab_size,
            embedding_dim=model_config['embedding_dim'],
            data_dir=paths_config.get('embeddings_dir', 'data')
        )

    # Build model
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)

    model = create_bilstm_model(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=model_config['embedding_dim'],
        sequence_length=data_config['sequence_length'],
        lstm_units=model_config['lstm_units'],
        dense_units=model_config['dense_units'],
        dropout_rates=model_config['dropout_rates'],
        embedding_matrix=embedding_matrix,
        trainable_embeddings=model_config.get('trainable_embeddings', True),
        learning_rate=training_config['initial_lr']
    )

    print_model_summary(model)

    # Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    model, history = train_bilstm_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=training_config['epochs'],
        model_save_path=paths_config['model_save_path'],
        log_dir=paths_config['log_dir']
    )

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    metrics = evaluate_model(model, val_dataset)

    # Save tokenizer
    import pickle
    tokenizer_path = paths_config['model_save_path'].replace('.h5', '_tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(preprocessor.tokenizer, f)
    print(f"\nTokenizer saved to {tokenizer_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {paths_config['model_save_path']}")
    print(f"Logs saved to: {paths_config['log_dir']}")
    print(f"Final validation accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Final validation perplexity: {metrics.get('perplexity', 0):.2f}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Bi-LSTM text generation model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/bilstm_config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()
    main(args.config)
