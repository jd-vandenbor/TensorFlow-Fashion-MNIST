# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# end of imports ---------------------------

# get the training and test images
fashion_mnist = keras.datasets.fashion_mnist  # import the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # load the dataset

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# flatten the images values from 255 to 1 and cast them as floats
train_images = train_images / 255.0
test_images = test_images / 255.0

# set up the structure of the model/(neural network)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # flatten the first array from 1D to 2D for input layer
    keras.layers.Dense(128, activation=tf.nn.relu),  # make second layer with 128 nodes input into a relu function
    keras.layers.Dense(10, activation=tf.nn.softmax)    # make output layer with softmax function summing up to 1
])

# compile the model/(neural network)
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# feed the NN training images
model.fit(train_images, train_labels, epochs=5)

# test the accuracy after training
test_loss, test_acc = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)
print('Test accuracy:', test_acc)
print(predictions[0])
print(np.argmax(predictions[0]))


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
fig = plt.figure(figsize=(2*2*num_cols, 2*num_rows))
fig.suptitle("Below are some test images fed to the neural network to test it's accuracy", fontsize=16)
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()