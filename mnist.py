# from mnist import MNIST

# def format_data(images, labels):
# 	data = []
# 	for index in range(0, len(labels)):
# 		input = np.array(images[index]) / 255
# 		output = np.zeros(10)
# 		output[labels[index]] = 1.0
# 		data.append([input, output])
# 	return data

# print('Loading Data...')
# mndata = MNIST('samples')
# images, labels = mndata.load_training()
# test_images, test_labels = mndata.load_testing()

# hdr_training_data = format_data(images, labels)
# test_hbr_training_data = format_data(test_images, test_labels)

# images, labels = zip(*hdr_training_data)

"""
Load the MNIST dataset into numpy arrays
Author: Alexandre Drouin
License: BSD


"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import helper
import chicken
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
X_train = mnist.train.images.reshape(-1, 28, 28)
y_train = mnist.train.labels

X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
y_test = mnist.test.labels

del mnist
def to_tensor(image):
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    return trans(image)
print(X_train[0], y_train[0])
X_train = chicken.data2d_to_grayscale(X_train)

inputs_out_directory = 'torch_inputs_mnist/'
labels_out_directory = 'torch_labels_mnist/'

# print('Saving images...')
# for i, image in enumerate(X_train):
# 	name = inputs_out_directory+'input_'+str(i)+'.pt'
# 	tensor = to_tensor(image)
# 	torch.save(tensor, name)

print('Saving labels...')
for i, label in enumerate(y_train):
	# print(label)
	name = labels_out_directory+'label_'+str(i)+'.pt'
	# tensor = to_tensor(label)
	tensor = torch.from_numpy(label)
	torch.save(tensor, name)
print('Done')