import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the MLP architecture
def create_mlp(input_size, hidden_size, output_size):
    model = keras.Sequential([
        layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)),
        layers.Dense(output_size)
    ])
    return model

# Define the training function
def train_mlp(X_train, y_train, num_epochs, batch_size, learning_rate):
    # Initialize the MLP
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    hidden_size = 50
    mlp = create_mlp(input_size, hidden_size, output_size)
    
    # Compile the model
    mlp.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    
    # Train the model
    history = mlp.fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=num_epochs,
                      verbose=1)
    
    return mlp

# Generate some sample data
X_train = tf.random.normal((1000, 200))
y_train = tf.random.normal((1000, 6))

# Train the MLP
mlp = train_mlp(X_train, y_train, num_epochs=100, batch_size=32, learning_rate=0.001)

# Use the trained MLP to make predictions on new data
X_test = tf.random.normal((10, 200))
y_pred = mlp.predict(X_test)
print(y_pred)
