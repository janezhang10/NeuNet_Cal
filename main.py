import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
from pprint import pprint
import pyfiglet

import chicken
from resnet_unet import *
from data_loader import get_PT_dataloaders, get_dataloaders, get_efficient_dataloaders, tensor_to_np
import re

def split(string):
	return re.findall(r"[\w']+|[.,!?;]", string)


activation_flag_strings = {
	'leaky': nn.LeakyReLU,
	'sigmoid': nn.Sigmoid,
	'relu': nn.ReLU,
	'railu': nn.ReLU,
	'tan': nn.ReLU,
	'tangent': nn.Tanh,
}

dimension_flag_strings = {
	'input',
	'dimensions',
	'resolution'
}

num_classes_flag_strings = {
	'classes',
	'class',
	'output'
}

model_type_flag_strings = {
	'segmentation': ResNetUNet,
	'semantic': ResNetUNet,
	'classifier':  ConvNet,
	'classification': ConvNet,
	'conv': ConvNet,
	'convolution': ConvNet,
	'forward': ConvNet,
}

num_layers_flag_strings = {
	'layer',
	'layers',
}

num_epochs_flag_strings = {
	'epoch',
	'epochs',
	'iteration',
	'iterations'
}

dataset_flag_strings = {
	'building': ('torch_inputs_buildings/', 'torch_labels_buildings/'),
	'digits': ('torch_inputs_mnist/', 'torch_labels_mnist/')
}

# WHITESPACE = r'\s+'
WHITESPACE = ' '

test = 'make me a segmentation network with input dimensions of 192 pixels, relu activation functions, and 4 output classes'
# test = 'make me a segmentation network with input dimensions of 192 pixels and 4 output classes'
test2 = 'create an image classifier with input resolution of 64, relu nonlinearities, and 3 output classes'



def get_specs(input_string):

	model_type = get_model_type(input_string)

	if model_type == ResNetUNet:
		dict_args = {
			'model_type': model_type,
			'activation': get_activation(input_string),
			'input_dim': get_input_dim(input_string),
			'num_classes': get_num_classes(input_string),
			'dataset': dataset_flag_strings['building']
		}
		
	elif model_type == ConvNet:
		dict_args = {
			'model_type': model_type,
			'activation': get_activation(input_string),
			'input_dim': get_input_dim(input_string),
			'num_classes': get_num_classes(input_string),
			'num_layers': get_num_layers(input_string),
			'dataset': dataset_flag_strings['digits']
		}


	train_args = {
		'num_epochs': get_num_epochs(input_string)
	}

	set_default_args(train_args, {'num_epochs': 100})

	set_default_args(dict_args, model_type.default_args)

	dict_args.update(train_args)
	return dict_args

def get_num_epochs(input_string):
	for string in num_epochs_flag_strings:
		if string in input_string:
			tokens = split(input_string)
			try:
				index_of_flag = tokens.index(string)
			except:
				continue
			num_epochs = first_number_to_left(tokens, index_of_flag)
			return num_epochs

def set_default_args(dict_to_mutate, default_args):
	for key, value in dict_to_mutate.items():
		if value is None:
			dict_to_mutate[key] = default_args[key]

def get_num_layers(input_string):
	for string in num_layers_flag_strings:
		if string in input_string:
			tokens = split(input_string)
			try:
				index_of_flag = tokens.index(string)
			except:
				continue
			num_layers = first_number_to_left(tokens, index_of_flag)
			return num_layers

def get_model_type(input_string):
	for string in model_type_flag_strings:
		if string in input_string:
			return model_type_flag_strings[string]

def get_activation(input_string):
	for string in activation_flag_strings:
		if string in input_string:
			return activation_flag_strings[string]

def get_input_dim(input_string):
	for string in dimension_flag_strings:
		if string in input_string:
			tokens = split(input_string)
			index_of_flag = tokens.index(string)
			dim = first_number_to_right(tokens, index_of_flag)
			return dim

def get_num_classes(input_string):
	for string in num_classes_flag_strings:
		if string in input_string:
			tokens = split(input_string)
			try:
				index_of_flag = tokens.index(string)
			except:
				continue
			num_classes = first_number_to_left(tokens, index_of_flag)
			return num_classes

def first_number_to_right(input_tokens, start_index): #TODO: what if string is sixty-eight
	number = None
	input_tokens = input_tokens[start_index:]
	for token in input_tokens:
		try:
			number = int(token)
			return number
		except ValueError:
			continue

def first_number_to_left(input_tokens, start_index):
	number = None
	input_tokens = input_tokens[:start_index][::-1]
	for token in input_tokens:
		try:
			number = int(token)
			return number
		except ValueError:
			continue

def johnny(string):
	args = get_specs(string)
	reprint('NeuNet')
	print('='*172)
	print(string.capitalize())
	print('='*172)
	print('')
	print_dict(args)
	main(args)

def print_dict(d):

	for key, val in d.items():
		print(key.capitalize() + ':', val)

def reprint(string):
	ascii_banner = pyfiglet.figlet_format(str(string))
	print(ascii_banner)


def main(dict_args):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# device = torch.device("cpu")
	print('Device:', device)
	print('\n')

	NUM_CLASSES = 4
	DIM = 192
	ACTIVATION = nn.ReLU
	NUM_POINTS = 10000000

	MODEL_PATH = 'ckpt.pth'
	NUM_EPOCHS = dict_args['num_epochs']
	LEARNING_RATE = 0.01
	LEARNING_RATE_DECAY = 0.01
	LEARNING_RATE_DECAY_STEP = 50
	BATCH_SIZE = 1

	# Use this UNTIL we can figure out an efficient label RGB -> N Channels transform
	# dataloaders, image_datasets = get_dataloaders(width_height=DIM,
	# 							  num_points=NUM_POINTS,
	# 							  batch_size=BATCH_SIZE)

	# Use this once we figure out an efficient label RGB -> N Channels transform
	# dataloaders, image_datsets = get_PT_dataloaders(width_height=DIM,
	# 							  num_points=NUM_POINTS,
	# 							  batch_size=BATCH_SIZE,
	# 							  input_directory='torch_inputs/',
	# 							  label_directory='torch_labels/')


	# dataloaders, image_datsets = get_efficient_dataloaders(width_height=DIM, \
	# 	num_points=NUM_POINTS, \
	# 	batch_size=BATCH_SIZE, \
	# 	input_directory='torch_inputs_overfit/',\
	# 	label_directory='torch_labels_overfit/')

	dataloaders, image_datsets = get_efficient_dataloaders(width_height=dict_args['input_dim'], \
		num_points=NUM_POINTS, \
		batch_size=BATCH_SIZE, \
		input_directory=dict_args['dataset'][0],
		label_directory=dict_args['dataset'][1])

	model_type = dict_args['model_type']
	model = model_type(dict_args).to(device)

	if NUM_EPOCHS > 0:

		model.load(MODEL_PATH)

		optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=LEARNING_RATE_DECAY_STEP, gamma=LEARNING_RATE_DECAY)

		model = train_model(model, optimizer_ft, exp_lr_scheduler, NUM_EPOCHS, dataloaders, device)


	model.load(MODEL_PATH)
	# model.eval()



	# data_iter = iter(dataloaders['val'])
	display_titles = []
	display_images = []
	for inputs, labels in dataloaders['val']:
		# inputs, labels = next(data_iter)
		inputs = inputs.to(device)
		labels = labels.to(device)

		predictions = model(inputs)

		metrics = defaultdict(float)
		epoch_samples = inputs.size(0)
		loss = calc_loss(predictions, labels, metrics)
		print_metrics(metrics, epoch_samples, 'val')

		predictions = F.sigmoid(predictions)

		import helper
		input_images = helper.input_tensors_to_colorimg(inputs)
		mask_images = helper.label_tensors_to_colorimg(labels)
		prediction_images = helper.label_tensors_to_colorimg(predictions)

		display_images += input_images + prediction_images
		display_titles += ['Input', 'Prediction']

	# chicken.display_all(display_images, titles=display_titles)
	# chicken.display_many(display_images[:6], columns=2, rows=3, titles=display_titles)
	chicken.display_many(display_images[6:], columns=2, rows=3, titles=['Inputs', 'Predictions', '', '', '', ''])


	# data_iter = iter(dataloaders['val'])
	# inputs, labels = next(data_iter)
	# # for inputs, labels in data_iter:
	# inputs = inputs.to(device)
	# labels = labels.to(device)

	# predictions = model(inputs)

	# metrics = defaultdict(float)
	# epoch_samples = inputs.size(0)
	# loss = calc_loss(predictions, labels, metrics)
	# print_metrics(metrics, epoch_samples, 'val')

	# predictions = F.sigmoid(predictions)

	# import helper
	# input_images = helper.input_tensors_to_colorimg(inputs)
	# mask_images = helper.label_tensors_to_colorimg(labels)
	# prediction_images = helper.label_tensors_to_colorimg(predictions)
	# chicken.display_all(input_images + prediction_images, titles=['Input', 'Prediction'])

if __name__ == '__main__':

	# voice1 = 'make me a segmentation network using the buildings dataset, relu nonlinearities, 4 output classes, input dimensions of 64, for 0 epochs'
	voice1 = 'make me an image segmentation U-Net with an input size of 192 pixels and 4 output classes, and use relu nonlinearities. Train the model on 25 epochs of the buildings dataset'
	johnny(voice1)
