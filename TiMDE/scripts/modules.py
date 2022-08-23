import os
import cv2
import csv
import json
import numpy as np

from math import log

import cv2

import tensorflow as tf
import tensorflow.keras.backend as K
sess = tf.compat.v1.Session()
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
'''
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

from pre_trained_unet.model_sem_bse import (
	preprocess_input,
	postprocess_output	
)

import random

# Used to check for (mostly) black images.
from scipy.linalg import norm

# Used to generate a unique ID for the image sets.
import uuid

from matplotlib import image

# Used to save lossless images.
from PIL import Image

# Used to process images and generate line profiles.
import skimage as sk
from skimage.measure import profile_line as line


def show(image):
	# load image using cv2....and do processing.
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	# as opencv loads in BGR format by default, we want to show it in RGB.
	plt.show()
	
def show2(image):
	'''
	An alternative to the method show above, in case it doesn't work.
	'''
	plt.imshow((cv2.cvtColor(image, cv2.COLOR_BGR2RGB) * 255).astype(np.uint8))
	plt.show()
	
def show3(image):
	'''
	Yet another alternative. This is used in cases where the image had to be
	turned into a list so that it could be stored in JSON.
	'''
	plt.imshow((cv2.cvtColor(image, cv2.COLOR_BGR2RGB) * 255).astype(np.uint8), interpolation='nearest')
	plt.show()
	
def remove_scale_bar(image):
	'''
	Removes scale bar from microscopy images. 
	I think the algorithm is relatively stable now.
	
	Assumptions: Scale bar is relatively bright in comparison
	to the image and same color.
	
	Note that this algorithm will have a difficult time with images
	taken at low accelerating voltages.
	
	-Rohan
	
	Inputs
	--------------
	image : array
		Original image.
		
	Outputs
	--------------
	image_cropped : array
		Image with the scale bar removed, hopefully.
	'''

	# First, threshold the image by taking the average grayscale
	# value of the first few rows (where the scale bar definitely
	# won't be), and use that as the threshold value.
	avgs = []
	for i in range(image.shape[0]//3):
		row_sum = np.sum(image[i,:])
		row_avg = row_sum/image.shape[1]
		avgs.append(row_avg)

	threshold_value = np.sum(avgs)//(image.shape[0]//3)


	# Next, blur the image to get rid of sharp features.
	# Then, we will open and close the image multiple times to remove
	# features not removed by the threshold process.

	# Opening the image multiple times will remove isolated features
	# within the actual image. The scale bar will remain resilient to the
	# opening process.

	# Closing the image multiple times will then make the scale bar more
	# prominent. As the noise from the top of the image has already been
	# removed, it will not make the microstructure more prevalent.
	blur = cv2.blur(image,(13,13))
	ret,thresh = cv2.threshold(image,threshold_value,255,cv2.THRESH_BINARY_INV)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (13,13), iterations=10)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (13,13), iterations=10)


	# Now, we are ready to remove the scale bar.
	running_average = 0
	running_averages_collection = []
	differences = []

	for i in range(thresh.shape[0]-1, thresh.shape[0]-200, -1):
		if i < thresh.shape[0] - 3:
			# Calculate an iterating sum between the last three values and average.
			iter_sum = norm(thresh[i,:]) + norm(thresh[i+1,:]) + norm(thresh[i+2,:])
			running_average = iter_sum / (3)
			running_averages_collection.append(running_average)

			# Take the difference in average grayscale value between the maximum
			# running average and the norm of the current row.
			# This assumes that the scale bar is now the brightest part of the image.
			difference = abs(norm(thresh[i,:]) - max(running_averages_collection))
			differences.append(difference)

	# Next, take the maximum difference of each row with the maximum running average
	# and compare it to each difference again. This effectively finds the index where
	# the maximum difference happened and crops the image there.
	# Keep in mind that this is not perfect, but it does seem a hell of a lot more
	# robust and stable than more fancy algorithms. It's not perfect, but I think
	# it is a good start.

	for i in range(thresh.shape[0]-1, 0, -1):
		if abs(norm(thresh[i,:]) - max(running_averages_collection)) >= max(differences):
			crop_index = i+1
			break
			
	image_cropped = image[0:crop_index+1, :]
	
	return image_cropped

def remove_scale_bar_2(image, scale_bar_height=35):
	'''
	Attempts to remove the scale bar by a constant value.
	
	Inputs:
	-------------
	scale_bar_height : int
		Height of the scale bar. Default is set to 35, but
		you can change it as you need to.
		
	Outputs:
	-------------
	image_cropped : image
		Cropped image.
	'''
	return image[:-scale_bar_height, :]

	
class Shuffler():
	def __init__(self, image, grid_size):
		'''
		Inputs
		--------------
		image : array
			Input image.
			
		grid_size : int
			To split the image into NxN tiles, type N here.
			Larger N leads to larger tiles.
		'''
		self.image = image
		self.shuffled_image = None
		
		self.grid_size = grid_size
		assert grid_size < max(self.image.shape[0], self.image.shape[1]), "grid_size must be smaller than the image."
		
		self.shuffle_image()
		
	def shuffle_image(self, number_col_shuffles = 10, number_row_shuffles = 10):
		'''
		'''
		self.x_dim = self.image.shape[0]
		self.y_dim = self.image.shape[1]

		self.shuffled_image = self.image

		check = None
		number_of_tiles = min(self.image.shape[0], self.image.shape[1]) // self.grid_size

		self.grid_dim_x = self.image.shape[0] // number_of_tiles
		self.grid_dim_y = self.image.shape[1] // number_of_tiles

		times = random.randint(20,50)
		for _ in range(times):
			row_or_col = random.randint(0,1)
			if row_or_col == 0:
				self._column_shuffle()
			elif row_or_col == 1:
				self._row_shuffle()
		
	def _column_shuffle(self):
		'''
		Document
		'''
		tiles = []
		shuffled_columns = []

		for y in range(0, self.y_dim, self.grid_dim_y):
			tiles = []
			shuffled_column = []

			for x in range(0, self.x_dim, self.grid_dim_x):
				tiles.append(self.shuffled_image[x:x+self.grid_dim_x, y:y+self.grid_dim_y])

			shuffled_tiles = tiles
			random.shuffle(shuffled_tiles)
			shuffled_column = np.vstack(shuffled_tiles)
			shuffled_columns.append(shuffled_column)

		self.shuffled_image = np.hstack(shuffled_columns)

	def _row_shuffle(self):
		'''
		'''
		tiles = []
		shuffled_columns = []

		for x in range(0, self.x_dim, self.grid_dim_x):
			tiles = []
			shuffled_column = []

			for y in range(0, self.y_dim, self.grid_dim_y):
				tiles.append(self.shuffled_image[x:x+self.grid_dim_x, y:y+self.grid_dim_y])

			shuffled_tiles = tiles
			random.shuffle(shuffled_tiles)
			shuffled_column = np.hstack(shuffled_tiles)
			shuffled_columns.append(shuffled_column)

		self.shuffled_image = np.vstack(shuffled_columns)

	
def generate_shuffled_image(image, percentage=0.03):
	'''
	Calls the Shuffler class.
	
	Inputs
	---------------
	image : array
		Original image.
		
	percentage : float
		Percentage of 1-D space of the ROI you'd like the
		grid to take up.
		0.5 splits the image into fourths, and is the upper bound.
	
	Outputs
	---------------
	shuffled_image : array
		Shuffled image.
	'''
	assert percentage <= 0.5 and percentage > 0, "percentage must be between 0 and 0.5."
	percentage = percentage**-1
	dim = int(min(image.shape[0], image.shape[1]) / percentage)
	shuffled_image = Shuffler(image, dim).shuffled_image
	
	return shuffled_image

def _get_sample_roi_dimensions(image, sample_percentage):
	'''
	Gets the dimensions of the ROI based on the image.
	
	ROI will be square.
	
	sample_length is calculated by dividing each dimension
	of the original image by sample_percentage, and taking the smaller
	of the two numbers.
	
	Inputs
	--------------
	image : array
		Original image.
		
	sample_percentage : int
		Changes the size of the ROI. Larger numbers mean larger ROI.
		
	Outputs
	--------------
	sample_length : int
		Length of the sides of the ROI region. Remember that the ROI
		is square.
	'''
	assert isinstance(sample_percentage, int)
	
	x_length = image.shape[0]
	y_length = image.shape[1]

	# I'd like the ROIs to be square, which is why I take the average of
	# the dimensions.
	x_sample_length = x_length // (100-sample_percentage)
	y_sample_length = y_length // (100-sample_percentage)

	sample_length = min(x_sample_length, y_sample_length)
	
	if sample_length % 2 != 0:
		x_sample_length -= 1
		if x_sample_length <= 0:
			print("This image is too small to be used for data augmentation!")
			
	return sample_length


def create_sample_ROI(image, sample_percentage=10, verbose=False):
	'''
	Create the center of the ROI by randomly selecting points on
	the x- and the y- axis. If the center is too close to the edge
	(meaning that the ROI will go off the page), the selection 
	trial is run again until a valid coordinate has been chosen.
	
	Inputs
	--------------
	image : array
		Original image.
		
	sample_percentage : int
		Changes the size of the ROI. Larger numbers mean larger ROI.
		Between 1-98. ROIs become exponentially more difficult to find
		as sample_percentage approaches 100. So 95 is the cutoff.
		
	Outputs
	--------------
	x_random : int
		Randomly selected x-coordinate for ROI.
		
	y_random : int
		Randomly selected y-coordinate for ROI.
	'''
	if sample_percentage <= 98 and sample_percentage >= 0:
		sample_length = _get_sample_roi_dimensions(image, sample_percentage)
	else:
		raise ValueError('input arg sample_length must be between 0 and 95.')
	x_length = image.shape[0]
	y_length = image.shape[1]
	
	x_random = random.randint(1, x_length+1)
	y_random = random.randint(1, y_length+1)

	while ( 
		(y_random + sample_length) > y_length
		or (y_random - sample_length) < 0 
	):
		if verbose:
			print(
				"Cannot create ROI: y_random exceeds ROI center limits.\n"
				"A new y_random will be chosen.\n\n"
			)
		y_random = random.randint(1, y_length+1)

	while ( 
		(x_random + sample_length) > x_length
		or (x_random - sample_length) < 0 
	):
		if verbose:
			print(
				"Cannot create ROI: x_random exceeds ROI center limits.\n"
				"A new x_random will be chosen.\n\n"
			)
		x_random = random.randint(1, x_length+1)
		
	roi = image[
		x_random - sample_length : x_random + sample_length,
		y_random - sample_length : y_random + sample_length
	]
		
	return roi


def _noisy(image, extra=1):
	'''
	Noise Generator.
	
	A type of noising method is randomly selected, and then applied 
	to the image.
	
	Inputs
	--------------
	image : array
		Image to be modified.
		
	extra : int
		Number of times to noise the image.
		
	Outputs
	--------------
	noisy : array
		Noised image.
	'''
	
	noisy = np.copy(image)
	
	for _ in range(int(extra)):
		noise_type_index = random.randint(0, 3)

		noise = ["gauss", "s&p", "poisson", "speckle"]

		noise_typ = noise[noise_type_index]

		if noise_typ == "gauss":
			row,col= image.shape
			mean = random.uniform(5, 20)
			var = random.uniform(5, 10)
			sigma = var**0.5
			gauss = np.random.normal(mean,sigma,(row,col))
			gauss = gauss.reshape(row,col)
			noisy = noisy + gauss
			noisy = np.array(noisy, dtype=np.uint8)

		elif noise_typ == "s&p":
			row,col = image.shape
			s_vs_p = random.uniform(0.3, 0.7)
			amount = random.uniform(0, 3)
			out = np.copy(noisy)

			# Salt mode
			num_salt = np.ceil(amount * image.size * s_vs_p)
			coords = [np.random.randint(0, i - 1, int(num_salt))
				  for i in image.shape]
			out[tuple(coords)] = 1

			# Pepper mode
			num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
			coords = [np.random.randint(0, i - 1, int(num_pepper))
				  for i in image.shape]
			out[tuple(coords)] = 0

			noisy = out
			noisy = np.array(noisy, dtype=np.uint8)

		elif noise_typ == "poisson":
			vals = len(np.unique(noisy))
			vals = 2 ** np.ceil(np.log2(vals))

			result = None
			while result is None:
				try:
					noisy = np.random.poisson(noisy * vals) / float(vals)
					result = 1
				except:
					 pass

			noisy = np.array(noisy, dtype=np.uint8)
	
		elif noise_typ =="speckle":
			multiplicative_factor = random.randint(1, 250)
			row,col = image.shape
			gauss = np.random.randn(row,col)
			gauss = gauss.reshape(row,col)		
			noisy = noisy + (noisy * gauss)**multiplicative_factor
			noisy = np.array(noisy, dtype=np.uint8)
			
	row,col = image.shape
	s_vs_p = random.uniform(0.3, 0.7)
	amount = random.uniform(0, 0.7)
	out = np.copy(noisy)

	# Salt mode
	num_salt = np.ceil(amount * image.size * s_vs_p)
	coords = [np.random.randint(0, i - 1, int(num_salt))
		  for i in image.shape]
	out[tuple(coords)] = 1

	# Pepper mode
	num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
	coords = [np.random.randint(0, i - 1, int(num_pepper))
		  for i in image.shape]
	out[tuple(coords)] = 0

	noisy = out
	noisy = np.array(noisy, dtype=np.uint8)
	
	noisy = _blur(noisy)
	noisy = _pixelate(noisy)
			
	return noisy

def _blur(image):
	'''
	Blurs the image with a kernel relative to the image's size.
	'''
	x = image.shape[1]
	y = image.shape[0]
	
	random_number = random.randint(x//1, x//0.3)
	
	kernel_x = x // random_number
	if kernel_x % 2 == 0:
		kernel_x += 1
		
	random_number = random.randint(y//1, y//0.3)

	kernel_y = y // random_number
	if kernel_y % 2 == 0:
		kernel_y += 1

	image_blurred = cv2.GaussianBlur(image,(kernel_x, kernel_y),0)

	return image_blurred

def _pixelate(image):
	# Get input size
	height, width = image.shape[:2]

	# Desired "pixelated" size
	w, h = (height//2, width//2)

	# Resize input to "pixelated" size
	temp = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

	# Initialize output image
	image_pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

	return image_pixelated
	
	
def flip(roi, mirror_plane):
	'''
	Flip the roi image.
	
	Inputs
	--------------
	roi : array
		Original randomly selected roi image.
		
	mirror_plane : int (0, -1, 1)
		plane that you'd like to flip on.
		0 is vertical, 1 is horizontal, -1 is both.
		
	original_roi_images : str
		Folder path to where you'd like the original augmented roi images stored.
		
	mask_images : str
		Folder path to where you'd like the noised roi images stored.
		
	Outputs
	--------------
	flip : array
		flipped image.
		
	results : dict
		Structure of the dictionary is defined below.
		The keys are the ids of the validation/training couple.
		The values are the keys of the respective image, "Validation"
		for the validation, clean images and "Train" for the training, 
		noised up images.
	'''

	if mirror_plane == 0:
		flip = cv2.flip(roi, mirror_plane)
		return flip
	elif mirror_plane == 1:
		flip = cv2.flip(roi, mirror_plane)
		return flip
	elif mirror_plane == -1:
		flip = cv2.flip(roi, mirror_plane)
		return flip
	else:
		raise ValueError("input arg mirror_plane must be 0, -1, or 1.")
		
def _rotate_bound(image, angle):
	'''
	From: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
	
	Properly rotates non-square images.
	'''
	# grab the dimensions of the image and then determine the
	# center
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	# grab the rotation matrix (applying the negative of the
	# angle to rotate clockwise), then grab the sine and cosine
	# (i.e., the rotation components of the matrix)
	M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY
	# perform the actual rotation and return the image
	return cv2.warpAffine(image, M, (nW, nH))

def rotate_right_angle(image, angle):
	'''
	Rotate an image at 90 degree angles. 0 and 360 are not allowed.
	
	Inputs
	--------------
	image : array
		Original image.
		
	Outputs
	--------------
	rotate : array
		Rotated image.
	'''
	if angle == 90:
		rotate = _rotate_bound(image, angle)
		rotate = np.array(rotate, dtype=np.uint8)
		return rotate
	elif angle == 180:
		rotate = _rotate_bound(image, angle)
		rotate = np.array(rotate, dtype=np.uint8)
		return rotate
	elif angle == 270:
		rotate = _rotate_bound(image, angle)
		rotate = np.array(rotate, dtype=np.uint8)
		return rotate
	else:
		raise ValueError("input arg angle must be 90, 180, or 270.")
		
def clean_image(image):
	'''
	Brightens up the image.
	'''
	blur = cv2.GaussianBlur(image, (7,7), 1)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	image = clahe.apply(blur)
	return image

def noise_routine(comparison_image, extra=1):
	'''
	Noises image as-per the private function _noisy.
	
	Inputs
	--------------
	comparison_image : array
		original image.
		
	extra : int
		number of extra times you'd like to noise the image.
		Default is 1!
		
	Outputs
	--------------
	training_image : array
		noised up image.
	'''
	assert (isinstance(comparison_image, np.ndarray)
			and len(comparison_image.shape) == 2), "Error: input argument comparison_image must be a 2-D array."
	
	image_threshold = norm(comparison_image)
	training_image = 0
	while norm(training_image) < image_threshold/2:
		training_image = _noisy(comparison_image, extra=1)

	return training_image
		
def save_training_data(folder_path, validation_image, training_image):
	'''
	Saves training data to the folder you specify.
	
	results is assumed to be a dict.
	
	Inputs
	--------------
	validation_image, training_image : arrays
		validation and training images to save to JSON.
	'''
	assert (isinstance(validation_image, np.ndarray)
			and len(validation_image.shape) == 2), "Error: input argument validation_image must be a 2-D array."
	
	assert (isinstance(training_image, np.ndarray)
			and len(training_image.shape) == 2), "Error: input argument validation_image must be a 2-D array."
	
	results = {}
	batch_id = str(uuid.uuid4())

	try:
		validation_image = Image.fromarray(validation_image)
		training_image = Image.fromarray(training_image)
		
		validation_path = os.path.join(folder_path, 'validation', batch_id + '.tiff')
		training_path = os.path.join(folder_path, 'train', batch_id + '.tiff')
		
		validation_image.save(validation_path)
		training_image.save(training_path)
		
	except Exception as e:
		print('Error encountered in save_training_data: {}'.format(e))
	
	return validation_path, training_path

def write_to_manifest(csv_file, training_path, validation_path):
	'''
	Creates a manifest for the training data and writes to it.
	
	Inputs
	--------------
	csv_file : str
		csv file name for the manifest
	
	training_path, validation_path : string
		validation and training image paths.
	'''
	with open(csv_file, 'a+', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([training_path, 'training'])
		writer.writerow([validation_path, 'validation'])
		
	return None


class Denoiser():
    def __init__(self, model_architecture_path, model_weights_path, scale_bar_height):
        self.model_architecture_path = model_architecture_path
        self.model_weights_path = model_weights_path
        self.scale_bar_height = scale_bar_height
        self.load_denoising_model()
        
    def remove_scale_bar_2(self, image, scale_bar_height):
    	'''
    	Attempts to remove the scale bar by a constant value.
    	
    	Inputs:
    	-------------
    	scale_bar_height : int
    		Height of the scale bar. Default is set to 35, but
    		you can change it as you need to.
    		
    	Outputs:
    	-------------
    	image_cropped : image
    		Cropped image.
    	'''
    	return image[:-scale_bar_height, :]

    def load_denoising_model(self):
        '''
        Loads the denoising model. Should be in a folder called 'pre_trained_unet'.

        Inputs:
        --------------
        model_architecture_path : str
            Should be a .json file containing the model architecture.

        model_weights_path : str
            Should be an .hdf5 file containing the model weights.

        Outputs:
        --------------
        model : Model object
            Returns the model. I don't really know what type it is.

        '''

        with open(self.model_architecture_path, 'r') as f:
            data = json.load(f)

        data_str = json.dumps(data)
        self.model = model_from_json(data_str)
        self.model.load_weights(self.model_weights_path)

        return self.model

    def _pad_power_two(self, dimension):
        '''
        An internal command to automatically find the number of black pixels
        to add to a dimension of the nearest power of 2 larger than dimension.
        
        Inputs:
        --------------
        dimension : int
            Input dimension of the image. Can be x or y.
            
        Outputs:
        --------------
        pad : int
            The nearest power of 2 larger than dimension.
        '''
        
        pad = 1
        while(pad < dimension):
            pad *= 2
        return pad

    def _predict(self, input_image):
        '''
        Denoise the image at input_path, and return the denoised image
        at output_path.
        
        Inputs:
        --------------
        input_image : array
            Input image.
        '''
        print('_predict')
        try:
            tensor = input_image.reshape([1, input_image.shape[0], input_image.shape[1], 1])
        except:
            tensor = input_image[:, :, 2].reshape([1, input_image.shape[0], input_image.shape[1], 1])
        
        tensor = np.tile(tensor, reps=3)
        image_in = preprocess_input(tensor)
        self.result = self.model.predict(image_in)
        self.result = self.result[0, :, :, 0][:self.input_image.shape[0], :self.input_image.shape[1]]
        print(self.result)
        return None
    
    def denoise_image(self, input_path, tile_dim_x = 2**10, tile_dim_y = 2**10):
        '''
        Because of memory issues, this function will process larger images in tiles.
        '''
        print('denoise_image')
        self.result = None
        self.input_image = None
        self.input_image = image.imread(input_path)
        self.input_image = np.array(self.input_image)
        # self.input_image = self.remove_scale_bar_2(self.input_image, scale_bar_height=self.scale_bar_height)
        
        # Pad the image so that it can fit properly inside the model.
        # -----------------------------------------------------------
        x_pad = self._pad_power_two(self.input_image.shape[0])
        y_pad = self._pad_power_two(self.input_image.shape[1])
        
        padded_image = cv2.copyMakeBorder(
            self.input_image, 
            0, x_pad - self.input_image.shape[0],
            0, y_pad - self.input_image.shape[1],
            cv2.BORDER_CONSTANT
        )
        # -----------------------------------------------------------
        
        x_dim = self.input_image.shape[0]
        y_dim = self.input_image.shape[1]

        pad_x_dim = padded_image.shape[0]
        pad_y_dim = padded_image.shape[1]

        new_image = np.zeros((padded_image.shape[0], padded_image.shape[1]))

        for x in range(0, x_dim, tile_dim_x):
            for y in range(0, y_dim, tile_dim_y):
                tile = []
                tile = padded_image[x:x+tile_dim_x, y:y+tile_dim_y]
                self._predict(tile)
                new_image[x:x+tile_dim_x, y:y+tile_dim_y] = self.result
                self.result = None
                
        self.result = new_image.astype(np.uint8)
        self.result = self.result[:self.input_image.shape[0], :self.input_image.shape[1]]