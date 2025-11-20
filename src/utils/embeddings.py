"""
Utility functions for loading and preparing pre-trained embeddings.
"""

import os
import numpy as np
import urllib.request
import zipfile


def download_glove_embeddings(embedding_dim=100, data_dir='data'):
    """
    Download GloVe embeddings if not already present.

    Args:
        embedding_dim: Dimension of embeddings (50, 100, 200, or 300)
        data_dir: Directory to save embeddings

    Returns:
        Path to the embeddings file
    """
    os.makedirs(data_dir, exist_ok=True)

    glove_file = f'glove.6B.{embedding_dim}d.txt'
    glove_path = os.path.join(data_dir, glove_file)

    if os.path.exists(glove_path):
        print(f"GloVe embeddings already exist at {glove_path}")
        return glove_path

    # Download GloVe
    print(f"Downloading GloVe {embedding_dim}d embeddings...")
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    zip_path = os.path.join(data_dir, 'glove.6B.zip')

    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(glove_file, data_dir)

        os.remove(zip_path)
        print(f"GloVe embeddings extracted to {glove_path}")

    except Exception as e:
        print(f"Error downloading GloVe embeddings: {e}")
        print("You can manually download from: http://nlp.stanford.edu/data/glove.6B.zip")
        return None

    return glove_path


def load_glove_embeddings(glove_path, embedding_dim=100):
    """
    Load GloVe embeddings from file.

    Args:
        glove_path: Path to GloVe embeddings file
        embedding_dim: Dimension of embeddings

    Returns:
        Dictionary mapping words to embedding vectors
    """
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings_index = {}

    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print(f"Loaded {len(embeddings_index)} word vectors")
        return embeddings_index

    except FileNotFoundError:
        print(f"GloVe file not found at {glove_path}")
        print("Embeddings will be trained from scratch.")
        return {}


def create_embedding_matrix(word_index, embeddings_index, embedding_dim, vocab_size):
    """
    Create embedding matrix from pre-trained embeddings.

    Args:
        word_index: Dictionary mapping words to indices
        embeddings_index: Dictionary mapping words to embedding vectors
        embedding_dim: Dimension of embeddings
        vocab_size: Vocabulary size

    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    found_count = 0

    for word, i in word_index.items():
        if i >= vocab_size:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_count += 1
        else:
            # Initialize with random values for words not in GloVe
            embedding_matrix[i] = np.random.normal(0, 0.1, embedding_dim)

    coverage = (found_count / len(word_index)) * 100
    print(f"Pre-trained embeddings coverage: {coverage:.2f}% ({found_count}/{len(word_index)} words)")

    return embedding_matrix


def prepare_pretrained_embeddings(word_index, vocab_size, embedding_dim=100, data_dir='data'):
    """
    Complete pipeline to prepare pre-trained embeddings.

    Args:
        word_index: Dictionary mapping words to indices
        vocab_size: Vocabulary size
        embedding_dim: Dimension of embeddings
        data_dir: Directory containing/for embeddings

    Returns:
        Embedding matrix
    """
    # Download if needed
    glove_path = download_glove_embeddings(embedding_dim, data_dir)

    if glove_path is None or not os.path.exists(glove_path):
        print("Using random initialization for embeddings")
        return np.random.normal(0, 0.1, (vocab_size, embedding_dim)).astype('float32')

    # Load embeddings
    embeddings_index = load_glove_embeddings(glove_path, embedding_dim)

    if not embeddings_index:
        print("Using random initialization for embeddings")
        return np.random.normal(0, 0.1, (vocab_size, embedding_dim)).astype('float32')

    # Create embedding matrix
    embedding_matrix = create_embedding_matrix(word_index, embeddings_index, embedding_dim, vocab_size)

    return embedding_matrix
