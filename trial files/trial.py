import pickle
import random

# Load the features from the pickle file
with open("data/all_spectra.pkl", "rb") as f:
    features = pickle.load(f)

# Load the labels from the pickle file
with open("data/surface_combinations.pkl", "rb") as f:
    labels = pickle.load(f)

# Combine the features and labels into a list of tuples
data = list(zip(features, labels))

# Shuffle the data randomly
random.shuffle(data)

print(data[0])