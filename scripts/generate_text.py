#!/usr/bin/env python3
"""
Text generation script using trained model.
"""

import os
import sys
import argparse
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from src.inference.generator import TextGenerator
from src.utils.config import load_config, get_inference_config


def main(model_path, tokenizer_path, prompt, num_words, temperature, top_k, num_samples):
    """
    Generate text using trained model.

    Args:
        model_path: Path to trained model
        tokenizer_path: Path to tokenizer
        prompt: Text prompt
        num_words: Number of words to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        num_samples: Number of samples to generate
    """
    print("="*60)
    print("TEXT GENERATION")
    print("="*60)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Create generator
    # Infer sequence length from model
    sequence_length = model.layers[0].input_shape[1]
    generator = TextGenerator(model, tokenizer, sequence_length=sequence_length)

    print(f"\nConfiguration:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Words to generate: {num_words}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Number of samples: {num_samples}")

    # Generate text
    print("\n" + "="*60)
    print("GENERATED TEXT")
    print("="*60)

    if num_samples == 1:
        text = generator.generate_text(
            prompt=prompt,
            num_words=num_words,
            temperature=temperature,
            top_k=top_k
        )
        print(f"\n{text}\n")
    else:
        samples = generator.generate_multiple(
            prompt=prompt,
            num_samples=num_samples,
            num_words=num_words,
            temperature=temperature,
            top_k=top_k
        )

        for i, text in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(text)
            print()

    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using trained model')
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
        '--prompt',
        type=str,
        default='To be or not to be',
        help='Text prompt'
    )
    parser.add_argument(
        '--num-words',
        type=int,
        default=100,
        help='Number of words to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to generate'
    )

    args = parser.parse_args()
    main(
        args.model,
        args.tokenizer,
        args.prompt,
        args.num_words,
        args.temperature,
        args.top_k,
        args.num_samples
    )
