"""Functions for tokenizing."""
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word: str):
    """Map part of speech tag to verb or noun.
    To reduce variability, adjectives and adverbs will be coded as verbs.
    This will turn "continued (adj) music (n)" into "continue (v) music (n)"
    
    References to part of speech codes:
    https://stackoverflow.com/questions/29332851/what-does-nn-vbd-in-dt-nns-rb-means-in-nltk
    
    Parameters
    ----------
    word : str
        single word

    Returns
    -------
    wordnet part of speech
        defaults to noun
    """
    # Get the first character of the part of speech code determined by nltk
    tag = nltk.pos_tag([word])[0][1][0].lower()
    return "v" if tag in ["j", "r", "v"] else "n"


def body_to_token(text: str) -> list:
    """Convert text body into tokenized words.
    Does the following:
    1. Change to lowercase.
    2. Use regex to clean text.
    3. Split string into array with 1 word per element.
    4. Remove stopwords (e.g. i, me, a, the).
    5. Reduce each word to its basic form according to its part of speech.
    
    Parameters
    ----------
    text : str
        body text -> "For example, this sentence. And this."

    Returns
    -------
    list
        list of tokens -> ["example", "sentence"]
    """
    # Lower text
    text = text.lower()
    # Clean text
    text = text.replace("\n", " ")  # Turn newlines into spaces
    text = text.replace(" u.s.", " usa")  # Turn all "u.s." into "usa", lemmatizer would turn "us" into "u" otherwise
    text = re.sub(r"\.", "", text)  # Remove periods, this will squash abbreviations ("l.p." -> "lp")
    text = re.sub(r"[0-9]", "", text)  # Remove numbers
    text = re.sub(r"[\W_]", " ", text)  # Turn any non-alphanumber character into space
    text = re.sub(r"\s+", r" ", text)  # Reduce continuous spaces
    # Split into words
    text = text.split()
    # Remove stopwords, including the word "reuter" which appears at the end of the documents
    text = [word for word in text if word not in ["reuter"] + stopwords.words("english")]
    # Lemmatize
    text = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text]
    return text
