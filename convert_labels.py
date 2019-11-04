import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import helper
import chicken
def to_tensor(image):
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    return trans(image)
BACKGROUND = np.array([0, 0, 170])
WALL = np.array([0, 0, 255])
DOOR = np.array([0, 170, 255])
WINDOW = np.array([0, 85, 255])

def simplify_label(label):
	label = np.array(label)
	width, height, _ = label.shape
	one_hot_label = np.zeros((width, height, 4))

	for x in range(width):
		for y in range(height):
			rgb = label[x][y]
			if np.allclose(rgb, WALL):
				one_hot_label[x][y][1] = 1
			elif np.allclose(rgb, DOOR):
				one_hot_label[x][y][2] = 1
			elif np.allclose(rgb, WINDOW):
				one_hot_label[x][y][3] = 1
			else:
				one_hot_label[x][y][0] = 1
	return one_hot_label


input_directory='inputs/'
label_directory='labels/'

inputs_out_directory = 'torch_inputs/'
labels_out_directory = 'torch_labels/'

num_points = 100000000
width_height = 192

preprocess_transform = transforms.Compose([
									transforms.Resize(width_height),
									transforms.CenterCrop(width_height),
									transforms.ToTensor()
									])

lazy_label_transform = transforms.Compose([
									transforms.ToPILImage(mode=None),
									transforms.Lambda(simplify_label),
									transforms.ToTensor()])


# test = torch.load(inputs_out_directory + 'input_0.pt')
# print(type(test))
# test = helper.input_tensors_to_colorimg([test])
# chicken.display_all(test)

# test = torch.load(labels_out_directory + 'label_0.pt')
# test = helper.label_tensors_to_colorimg([test])
# chicken.display_all(test)


# inputs = list(zip(*datasets.ImageFolder(root=input_directory, transform=preprocess_transform)))[0][:num_points]
# labels = list(zip(*datasets.ImageFolder(root=label_directory, transform=lazy_label_transform)))[0][:num_points]










# inputs = chicken.get_images('inputs/data/')[:num_points]
# inputs = chicken.resize_and_smart_crop_square(inputs, width_height)

labels = chicken.get_images('labels/data/')[:num_points]
labels = chicken.resize_and_smart_crop_square(labels, width_height)

# print('Saving images...')
# for i, image in enumerate(inputs):
# 	name = inputs_out_directory+'input_'+str(i)+'.pt'
# 	tensor = to_tensor(image)
# 	torch.save(tensor, name)

print('Saving labels...')
for i, label in enumerate(labels):
	name = labels_out_directory+'label_'+str(i)+'.pt'
	simplified = simplify_label(label)
	tensor = to_tensor(simplified)
	torch.save(tensor, name)
print('Done')