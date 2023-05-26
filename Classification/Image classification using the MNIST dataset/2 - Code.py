import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist.load_data()

# Split the dataset into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist

# Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Define the hyperparameter space to search over
learning_rates = [0.01, 0.001, 0.0001]
hidden_neurons = [64, 128, 256]

# Perform grid search over the hyperparameter space
best_accuracy = 0
best_lr = 0
best_neurons = 0

for lr in learning_rates:
    for neurons in hidden_neurons:
        # Define the model architecture with the current hyperparameters
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(neurons, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        # Compile the model with the current hyperparameters
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Train the model with the current hyperparameters
        history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(x_test, y_test)

        # Update the best hyperparameters if the current model is better
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_lr = lr
            best_neurons = neurons

# Print the best hyperparameters
print("Best learning rate: ", best_lr)
print("Best number of neurons: ", best_neurons)

# Evaluate the performance of the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Print the test accuracy
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

# Make predictions on the test set
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Plot a confusion matrix to visualize the performance of the model
confusion_matrix = tf.math.confusion_matrix(y_test, y_pred)
plt.imshow(confusion_matrix, cmap='binary')
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.title("Confusion matrix")
plt.colorbar()

# Print the classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)
