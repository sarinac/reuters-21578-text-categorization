"""Functions for generating bag of words."""
from collections import Counter
import numpy as np


def generate_bow(tokens_list: list, max_vocab=10000) -> dict:
    """Convert tokenized words into bag of words.
    This will be used to keep track of word count in the train data.
    
    Parameters
    ----------
    tokens_list : list
        list of token lists -> [["example", "sentence"], ["apple"]]
    max_vocab: int, default 10000
        maximum number of words in bag of words

    Returns
    -------
    dict
        dict of {token: frequency} in descending order
    """
    frequency = Counter()
    for tokens in tokens_list:
        frequency.update(tokens)
    print(f"""
        The most common words from {len(frequency)} documents are 
        {", ".join([f"{word[0]} ({word[1]})" for word in frequency.most_common(5)])}.
    """)
    frequency = dict(sorted(frequency.items(), key=lambda item: -item[1]))
    # Truncate to max vocab size
    original_length = len(frequency)
    frequency = dict(list(frequency.items())[:min(max_vocab, len(frequency))])
    print(f"Truncated word dictionary from {original_length} to {min(max_vocab, len(frequency))} words.")
    return frequency


def encode_and_pad(word_dict, sentence, pad=800):
    """Convert a sentence into a padded vector of numerical encodings.
    
    Parameters
    ----------
    word_dict : dictionary
        word dictionary for bag of words
    sentence : str
        body text
    pad : int
        size of vector (default=800)

    Returns
    -------
    list
        vectorized sentence
    """
    NOWORD = 0 # 0 = no word in document
    INFREQ = 1 # 1 = word in document but not in word dictionary
    
    working_sentence = [NOWORD] * (pad)

    for word_index, word in enumerate(sentence[:pad]):
        working_sentence[word_index] = word_dict.get(word, INFREQ)
    
    return working_sentence


def vectorize_data(word_dict, data, pad=800):
    """Convert all sentences in the dataset into a padded vector 
    of numerical encodings. The length of the sentence will be 
    the 0th element and the sentence will start on the 1st element.
    
    Parameters
    ----------
    word_dict : dictionary
        word dictionary for bag of words
    data : list
        list of sentences (list of words)
    pad : int
        size of vector (default=800)

    Returns
    -------
    np.array
        numpy array (matrix) of each processed sentence
    """
    result = []
    for sentence in data:
        vectorized_sentence = encode_and_pad(word_dict, sentence, pad)
        result.append(vectorized_sentence)
        
    return np.array(result)
