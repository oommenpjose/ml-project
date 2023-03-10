import pickle
import random

# Load the features from the pickle file
with open("data/all_spectra.pkl", "rb") as f:
    features = pickle.load(f)

# Load the labels from the pickle file
with open("data/surface_combinations.pkl", "rb") as f:
    labels = pickle.load(f)

# Combine the features and labels into a list of tuples
data = list(zip(features, labels[1]))

# Shuffle the data randomly
random.shuffle(data)

# Calculate the split index for the training/test split
split_idx = int(len(data) * 0.8)

# Split the data into training and test sets
train_data = data[:split_idx]
test_data = data[split_idx:]

# Save the training and test data to pickle files
with open("train_data.pkl", "wb") as f:
    pickle.dump(train_data, f)

with open("test_data.pkl", "wb") as f:
    pickle.dump(test_data, f)
