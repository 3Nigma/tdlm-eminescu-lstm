import numpy as np
import json

# Load the training data and prepare the dictionaries
with open('eminescu.txt', 'r') as f:
    text=f.read()
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

if __name__ == "__main__":
  print(json.dumps(int_to_vocab))
