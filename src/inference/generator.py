"""
Text generation module for inference.
"""

import numpy as np
import tensorflow as tf


class TextGenerator:
    """Text generator for trained models"""

    def __init__(self, model, tokenizer, sequence_length=20):
        """
        Initialize text generator.

        Args:
            model: Trained Keras model
            tokenizer: Fitted tokenizer
            sequence_length: Length of input sequences
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.word_index = tokenizer.word_index
        self.index_word = tokenizer.index_word

    def predict_next_word(self, input_text, temperature=1.0):
        """
        Predict the next word given input text.

        Args:
            input_text: Input text string
            temperature: Sampling temperature (higher = more random)

        Returns:
            Predicted word
        """
        # Tokenize input
        tokens = self.tokenizer.texts_to_sequences([input_text.lower()])[0]

        # Pad or truncate to sequence length
        if len(tokens) > self.sequence_length:
            tokens = tokens[-self.sequence_length:]
        elif len(tokens) < self.sequence_length:
            tokens = [0] * (self.sequence_length - len(tokens)) + tokens

        # Prepare input
        input_seq = np.array([tokens])

        # Predict
        predictions = self.model.predict(input_seq, verbose=0)[0]

        # Apply temperature
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        # Sample from distribution
        predicted_idx = np.argmax(predictions)
        predicted_word = self.index_word.get(predicted_idx, '<UNK>')

        return predicted_word

    def generate_text(
        self,
        prompt,
        num_words=50,
        temperature=0.8,
        top_k=40
    ):
        """
        Generate text with top-k sampling.

        Args:
            prompt: Initial text prompt
            num_words: Number of words to generate
            temperature: Sampling temperature
            top_k: Number of top words to sample from

        Returns:
            Generated text
        """
        generated_text = prompt
        current_text = prompt

        for _ in range(num_words):
            # Tokenize current text
            tokens = self.tokenizer.texts_to_sequences([current_text.lower()])[0]

            # Use last sequence_length tokens
            if len(tokens) > self.sequence_length:
                tokens = tokens[-self.sequence_length:]
            elif len(tokens) < self.sequence_length:
                tokens = [0] * (self.sequence_length - len(tokens)) + tokens

            # Prepare input
            input_seq = np.array([tokens])

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
            next_word = self.index_word.get(next_token, '<UNK>')

            # Append to generated text
            generated_text += ' ' + next_word
            current_text += ' ' + next_word

        return generated_text

    def generate_multiple(
        self,
        prompt,
        num_samples=3,
        num_words=50,
        temperature=0.8,
        top_k=40
    ):
        """
        Generate multiple text samples.

        Args:
            prompt: Initial text prompt
            num_samples: Number of samples to generate
            num_words: Words per sample
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            List of generated texts
        """
        samples = []

        for i in range(num_samples):
            print(f"\nGenerating sample {i+1}/{num_samples}...")
            text = self.generate_text(
                prompt=prompt,
                num_words=num_words,
                temperature=temperature,
                top_k=top_k
            )
            samples.append(text)

        return samples


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None], dtype=tf.int32)
])
def fast_predict(model, inputs):
    """
    Fast prediction function using TF graph optimization.

    Args:
        model: Keras model
        inputs: Input sequences

    Returns:
        Predictions
    """
    return model(inputs, training=False)
