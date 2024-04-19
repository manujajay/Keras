# Keras

This repository contains examples and guides on how to use Keras, a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation.

## Prerequisites

- Python 3.6 or higher
- pip package manager

## Installation

To use Keras, you should have TensorFlow installed as its backend. Follow these steps to set up Keras and TensorFlow:

1. Create a virtual environment:

   ```bash
   python -m venv keras-env
   source keras-env/bin/activate  # On Windows use `keras-env\Scripts\activate`
   ```

2. Install TensorFlow and Keras:

   ```bash
   pip install tensorflow
   pip install keras
   ```

## Example - Simple Sequential Model

Here's a simple example of defining a sequential model in Keras to classify handwritten digits from the MNIST dataset.

### `sequential_model.py`

```python
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
import keras

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
```

## Contributing

If you'd like to contribute to the development of the Keras examples, please fork this repository, create a new branch for your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
