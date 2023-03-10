import pickle
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

# Load the training data from pickle file
with open("train_data.pkl", "rb") as f:
    train_data = pickle.load(f)

# Load the test data from pickle file
with open("test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

# Separate the features and labels in the training data
train_X = [sample[0] for sample in train_data]
train_y = [sample[1] for sample in train_data]

# Separate the features and labels in the test data
test_X = [sample[0] for sample in test_data]
test_y = [sample[1] for sample in test_data]

# Create a support vector regression model
svr = SVR(kernel='linear')

# Create a multi-output regression model with the SVM model as the base estimator
model = MultiOutputRegressor(svr)

# Fit the model to the training data
model.fit(train_X, train_y)

# Use the trained model to predict the labels for the test data
pred_y = model.predict(test_X)

# Calculate the mean squared error between the predicted and actual labels
mse = mean_squared_error(test_y, pred_y)

print("Mean squared error:", mse)
