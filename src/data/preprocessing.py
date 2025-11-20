"""
Data preprocessing module for Shakespeare text generation.
Handles text loading, cleaning, tokenization, and sequence generation.
"""

import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ShakespeareDataPreprocessor:
    """Preprocessor for Shakespeare text data"""

    def __init__(self, vocab_size=10000, sequence_length=20, oov_token='<UNK>'):
        """
        Initialize the preprocessor.

        Args:
            vocab_size: Maximum vocabulary size
            sequence_length: Length of input sequences
            oov_token: Out-of-vocabulary token
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.oov_token = oov_token
        self.tokenizer = None
        self.text_data = None

    def load_text(self, file_path):
        """
        Load text from file.

        Args:
            file_path: Path to text file

        Returns:
            Loaded text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.text_data = f.read()
            print(f"Loaded {len(self.text_data)} characters from {file_path}")
            return self.text_data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

    def create_tokenizer(self, text=None):
        """
        Create and fit tokenizer on text.

        Args:
            text: Text to fit tokenizer on (uses self.text_data if None)

        Returns:
            Fitted tokenizer
        """
        if text is None:
            text = self.text_data

        if text is None:
            raise ValueError("No text data available. Call load_text() first.")

        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token=self.oov_token,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True
        )

        self.tokenizer.fit_on_texts([text])

        # Update vocab_size to actual size
        self.vocab_size = len(self.tokenizer.word_index) + 1

        print(f"Tokenizer created with vocabulary size: {self.vocab_size}")
        return self.tokenizer

    def text_to_sequences(self, text=None):
        """
        Convert text to sequences of integers.

        Args:
            text: Text to convert (uses self.text_data if None)

        Returns:
            List of integer sequences
        """
        if text is None:
            text = self.text_data

        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call create_tokenizer() first.")

        # Tokenize by words
        words = re.findall(r'\b\w+\b', text.lower())
        sequences = self.tokenizer.texts_to_sequences([' '.join(words)])

        return sequences[0]

    def create_training_sequences(self, text_sequences):
        """
        Create input/output sequence pairs for training.

        Args:
            text_sequences: List of integer tokens

        Returns:
            Tuple of (input_sequences, target_tokens)
        """
        input_sequences = []
        target_tokens = []

        for i in range(len(text_sequences) - self.sequence_length):
            input_seq = text_sequences[i:i + self.sequence_length]
            target = text_sequences[i + self.sequence_length]

            input_sequences.append(input_seq)
            target_tokens.append(target)

        print(f"Created {len(input_sequences)} training sequences")

        return np.array(input_sequences), np.array(target_tokens)

    def prepare_data(self, file_path, train_split=0.8):
        """
        Complete data preparation pipeline.

        Args:
            file_path: Path to text file
            train_split: Fraction of data for training

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        # Load and tokenize
        self.load_text(file_path)
        self.create_tokenizer()
        text_sequences = self.text_to_sequences()

        # Create sequences
        X, y = self.create_training_sequences(text_sequences)

        # Split data
        split_idx = int(len(X) * train_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"Training set: {len(X_train)} sequences")
        print(f"Validation set: {len(X_val)} sequences")

        return X_train, y_train, X_val, y_val

    def get_word_index(self):
        """Get word to index mapping"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer.word_index

    def get_index_word(self):
        """Get index to word mapping"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer.index_word
