import random
import re

def preprocess_text(text):
    """
    Preprocess the text by converting it to lowercase and tokenizing
    using a regular expression (this avoids the need for NLTK's punkt data).
    """
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def build_markov_chain(words, n=2):
    """
    Build a Markov chain from a list of words using 'n'-grams as keys.
    """
    markov_chain = {}
    for i in range(len(words) - n):
        key = tuple(words[i:i+n])
        next_word = words[i+n]
        if key not in markov_chain:
            markov_chain[key] = []
        markov_chain[key].append(next_word)
    return markov_chain

def generate_text(markov_chain, length=50):
    """
    Generate text using the Markov chain for a given number of words.
    """
    key = random.choice(list(markov_chain.keys()))
    result = list(key)
    
    for _ in range(length):
        next_words = markov_chain.get(key)
        if not next_words:
            break
        next_word = random.choice(next_words)
        result.append(next_word)
        key = tuple(result[-len(key):])
    return " ".join(result)

if __name__ == '__main__':
    text_data = (
        "Markov chains are a powerful tool for text generation. "
        "They use the probabilities of word sequences to generate new text. "
        "This simple example demonstrates how to build and use a Markov chain. "
        "Feel free to replace this sample text with any text of your choice."
    )
    
    words = preprocess_text(text_data)
    markov_chain = build_markov_chain(words, n=2)
    
    generated_text = generate_text(markov_chain, length=30)
    
    print("Generated Text:")
    print(generated_text)
