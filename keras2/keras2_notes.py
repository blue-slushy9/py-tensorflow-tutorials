# This model trains a neural network model to classify images of clothing

import tensorflow as tf

# Helper libraries

# NumPy is used for scientific calculations
import numpy as np
# MatPlotLib is a library for creating static, animated, and interactive
# visualizations; PyPlot is an interface layer built on top of MPL, it
# provides a collection of functions that allows you to work with MPL in
# a more MATLAB-like style, i.e. simpler and more concise ways to create plots
import matplotlib.pyplot as plt

print(tf.__version__)

# Import fashion dataset from MNIST from TF, assign to variable
fashion_mnist = tf.keras.datasets.fashion_mnist

# The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255;
# the labels are an array of integers, from 0 to 9, with each number
# corresponding to the class of clothing the image represents;
# load_data() is able to separate the dataset into training and testing sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Each image is mapped to a single label, since the class names are not
# included with the dataset, store them here to use later when plotting the
# images;
# '\' is a line break, needed to break up list into two lines for readability
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', \
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the format of the dataset before training the model, this command
# shows there are 60,000 images;
# In Keras, the .shape attribute is used to access the dimensional size (shape)
# of a tensor or layer output, it returns a tuple of integers representing the
# number of elements along each dimension of the tensor	 
train_images.shape

# This command shows there are 60,000 labels in the training set
len(train_labels)

# As stated above, each label is an integer between 0 and 9
train_labels

# There are 10,000 images in the test set, each image is 28x28 pixels
test_images.shape

# The test set contains 10,000 image labels
len(test_labels)

# Preprocess the data, this must be done before training the network;
# figure() creates a new figure object, which is the basic container for plots
# in Matplotlib; it creates a new figure window for plotting, a canvas where you
# can generate your plots and graphs; it also allows you to activate an existing
# figure if you provide a specific identifier;
plt.figure()
# imshow() displays an image as a two-dimensional data plot, it takes your image
# data and translates it into colors on a grid; it accepts an image represented
# as a 2D or 3D NumPy array; it does not directly return an image, rather it creates
# an AxesImage object, which represents the plotted image on the current figure---
# you can typically use plt.show() afterward to display the image on the screen;
plt.imshow(train_images[0])
# Adds a colorbar to the plot, which acts a legend that visually represents the color
# mapping used in the plot; many plots in MPL, like heatmaps and contour plots, rely on
# colormaps to translate data values into colors; the colorbar then provides a visual
# reference to for understanding this mapping as it has labels underneath the colors;
plt.colorbar()
# grid() controls the visibility and appearance of grid lines on your plots, which are
# used to provide reference points along the axes; in this case, grid lines are not enabled
# on our plot, as the argument is False
plt.grid(False)
# Finally, show() is used to display the plot we have created
plt.show()

# Scale values to a range of 0 to 1 before feeding them to the NN model---
# to do this, divide the values by 255; it's important that the training set
# and the testing set be preprocessed in the same way;
# We scale the values for improved gradient descent optimization, activation function behavior,
# standardization and normalization;
train_images = train_images / 255.0
test_images = test_images / 255.0

# To verify that the data is in the correct format and that you're ready to
# build and train the NN, display the first 25 images from the training set
# and display the class name below each image;
# figsize() controls the dimensions (width and height) of the figure window
# in inches, which allows you to customize the canvas where your plots are
# created---in this case, 10 inches wide and 10 inches high;
plt.figure(figsize=(10,10))
for i in range(25):
	# In this usage, subplot() is used to create smaller plots within a single
    # figure window, arranged in a grid-like layout; the syntax is as follows:
	# (nrows, ncols, index), where nrows is the number of rows in subplot grid,
    # ncols is the number of columns, and index is the index of the subplot within
	# the grid, starting from 1 by default; 
    plt.subplot(5,5,i+1)
	# xticks() controls the x-axis ticks and tick labels on your plots, which are
    # small marks along the axis that provide reference points for the data values;
	# providing an empty list as argument means to remove all tick marks and labels
    # from the x-axis
	plt.xticks([])
	# Same as xticks(), except for the y-axis
	plt.yticks([])
	plt.grid(False)
	# The second argument is optional and can be used to represent the intensity values
    # in the grayscale image; by default, MPL uses the viridis colormap;
	# 'cmap' means colormap, the value 'plt.cm' is the MPL colormap library, and 'binary'
    # is a built-in black-and-white colormap in that library 
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	# xlabel() sets a label for the x-axis, it takes a single argument that represents the
    # text you want to appear on the x-axis; in this case, the training labels
	plt.xlabel(class_names[train_labels[i]])
plt.show()

# BUILD THE MODEL

# Set up the layers: basic building block of a NN is the layer; layers extract
# representations from the data fed into them; most of deep learning consists
# of chaining together simple layers; most layers (e.g. tf.keras.layers.Dense)
# have parameters that are learned during training
model = tf.keras.Sequential([
    # Flatten transforms the format of the images from a two-dimensional array
	# (28x28 pixels) to a one-dimensional array (28*28 = 784 pixels), you can
	# think of this layer as unstacking rows of pixels in the image and lining
	# them up; this layer has no parameters to learn, it only reformats data
    tf.keras.layers.Flatten(input_shape=(28, 28)),
	# Dense layers are densely connectioned, or fully connected, neural layers;
	# this first Dense layer has 128 nodes (i.e. neurons); these Dense layers
	# return a logits array with a length of 10; each node contains a score
	# that indicates the current image belongs to one of the 10 classes
	tf.keras.layers.Dense(128, activation='relu'),
	# This Dense layer has 10 nodes
	tf.keras.layers.Dense(10)
])

# COMPILE THE MODEL

# Before the model is ready for training, it needs a few more settings---
# these are added during the model's compile step; OPTIMIZER: this is how the
# model is updated based on the data it sees and its loss function;
# LOSS FUNCTION: measures how accurate the model is during training, you want
# to minimize this function to "steer" the model in the right direction;
# METRIC: used to monitor the training and testing steps, the below example
# uses ACCURACY, the fraction of the images that are correctly classified
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			  metrics=['accuracy'])

# TRAIN THE MODEL

# Training the NNM requires the following steps:
# 1) Feed the training data to the model, e.g. train_images and train_labels
# 2) The model learns to associate images and labels
# 3) Ask the model to make predictions about a test set, e.g. the test_images
#    array
# 4) Verify that the predictions match the labels from the test_labels array

# FEED THE MODEL

# To start training, call the model.fit method (it is called this because it
# "fits" the model to the training data); train_images are the inputs, train_labels
# are what the model is trying to learn to predict, and epochs is the number of times
# times the entire training dataset is passed through by the model
model.fit(train_images, train_labels, epochs=10)

# EVALUATE ACCURACY

# Compare how the model performs on the test dataset;
# test_loss stores the loss value on the test dataset, test_acc stores the
# accuracy value; model.evaluate() is similar to model.fit() above, except
# that we are now in the testing phase; verbose=2 is the highest value for 
# output verbosity
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# RESULTS: if you run the program, you will see that the accuracy on the test
# dataset is a little less than the accuracy on the training dataset; this gap
# between training and test accuracy represents OVERFITTING; this happens when
# a ML model performs on new, previously unseen inputs than it does on the
# training data; an OVERFITTED MODEL "memorizes" the noise and details in the
# training dataset to a point where it negatively impacts the performance of
# the model on the new data

# MAKE PREDICTIONS

# Now that the model is trained, you can start using it to make predictions
# about some images; attach a softmax layer to convert the model's LINEAR
# OUTPUTS (i.e. LOGITS) to probabilities, which are often easier to interpret
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# Call the predict() method
predictions = probability_model.predict(test_images)										

# View the first prediction the model has made; when you run the program, you
# will see that a prediction is an array of 10 numbers, which represent the
# model's "confidence" that the image corresponds to each of the 10 different
# articles of clothing (i.e. classes)
predictions[0]

# View which label has the highest confidence value;
# argmax() is used to find the index, or indices, of the maximum value(s) along
# a specified axis in a NumPy array; it takes an array as its required argument
np.argmax(predictions[0])

# You can cross-reference the test_labels to verify the prediction
test_labels[0]

# Define functions to graph the full set of 10 class predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	# imshow() accepts an image represented as a 2D or 3D NumPy array
	plt.imshow(img, cmap=plt.cm.binary)
	# argmax() will identify index/indices of maximum value(s) in 
    # predictions_array
	predicted_label = np.argmax(predictions_array)
	# If the prediction is accurate...
	if predicted_label == true_label:
	    color = 'blue'
    else:
	    color = 'red'
	# Constructs a formatted string that will be used as the x-label for our
    # plot, said label will combine information about the predicted class and
	# its corresponding confidence level; {} is a placeholder for values that 
    # will be inserted into the string at specific positions; ':2.0 specifies 
	# a minimum field width of 2 and precision of 0 decimal places, i.e. an integer;
    # 'f' indicates the value should be formatted as a floating-point number; 
	# '%' will be included literally within the label string to make it a percentage;
    # '({})' is another placeholder for the predicted class name; '.format()' calls
	# the format method on the f-string literal, the arguments within the parentheses
    # will be inserted into the corresponding placeholders mentioned previously;
	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
	# max() will return the element with the highest value; we multiply the value by
    # 100 in order to make it a percentage
	100*np.max(predictions_array),
	class_names[true_label]),
	color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
	plt.grid(False)
	# range(10) will generate 0-9 as it is non-inclusive, which will in turn be used
    # by xticks() to label the x-axis
	plt.xticks(range(10))
	plt.yticks([])
	# bar() creates bar chart or bar plots, it visualizes categorical data using
    # rectangular bars with heights proportional to the values you provide; the
	# arguments are x, which is the list or array representing the positions of the
    # bars on the x-axis; height, which is a list or array containing the heights of
	# the bars, the length of this list should match x; color is optional but is used
    # to define the color(s) of the bars 
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	# ylim() acts as a setter and updates the y-axis limits of the current plot, in this
    # case with 0 as the bottom and 1 as the top
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)
	
	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

# VERIFY PREDICTIONS

# Verify 0th image, predictions, and prediction array; correct prediction
# labels are blue and incorrect prediction labels are red; the number gives
# the percentage (out of 100) for the predicted label
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Verify 12th image...
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels;
# Color correct predictions in blue and incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
# Not sure why we have to multiply all of these values, but I'll probably find
# out when I look at the output again
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
	plot_image(i, predictions[i], test_labels, test_images)
	plt.subplot(num_rows, 2*num_cols, 2*i+2)
	plot_value_array(i, predictions[i], test_labels)
# tight_layout() automatically adjusts the spacing around subplots in a figure
# to minimize wasted space; it is particularly useful when you have multiple
# subplots on a figure
plt.tight_layout()
plt.show()

# USE THE TRAINED MODEL

# Finally, use the trained model to make a prediction about a single image

# Grab an image from the test dataset
img = test_images[1]

print(img.shape)

# tf.keras models are optimized to make predictions on a batch, or collection,
# of examples at once; accordingly, even though you are using only a single
# image, you need to add it to a list

# Add the image to a batch where it's the only member;
# expand_dims() inserts a new dimension (axis) into an existing array, which is
# useful when you want to perform operations on arrays that require a specific
# number of dimensions; 'img' is the array that will be modified, 0 is the integer
# that specifies the position where the new dimension will be inserted
img = (np.expand_dims(img,0))

print(img.shape)

# Now predict the correct label for this image
predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
# '_' is a placeholder variable, which tells Python to execute the function
# but to discard the return value, it is commonly used when the function
# doesn't return a value you need to store or use explicitly; 'rotation=45'
# rotates the labels 45 degrees to prevent overlapping, which could easily
# occur as some of the class names are long
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# tf.keras.Model.predict returns a list of lists---one for each image in the
# batch of data; grab the predictions for our only image in the batch
np.argmax(predictions_single[0])