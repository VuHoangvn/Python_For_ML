import numpy as np 
from functools import partial
import PIL.Image
import tensorflow as tf 
import urllib.request
import os
import zipfile

def main():
	# Step 1 - download google's pre-trained neural network
	url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
	data_dir = 'download'
	model_name = os.path.split(url)[-1]
	local_zip_file = os.path.join(data_dir, model_name)
	if not os.path.exists(local_zip_file):
		# download
		model_url = urllib.request.urlopen(url)
		with open(local_zip_file, 'wb') as output:
			output.write(model_url.read())

		# extract
		with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
			zip_ref.extractall(data_dir)




	model_fn = 'tensorflow_inceptio_graph.pb'

	# Step 2 - Creating Tensorflow session and loading the model_url
	graph = tf.Graph()
	sess = tf.InteractiveSession(graph=graph)
	with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	t_input = tf.placeholder(np.float32, name='input') # define input tensor
	imagenet_mean = 117.0
	t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
	tf.import_graph_def(graph_def, {'input': t_preprocessed})

	layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
	feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]


	def render_deepdream(t_obj, img0=img_noise, iter_n = 10, step=1.5, octave_n=4, octave_scale=1.4):
		t_score = tf.reduce_mean(t_obj) #defining optimization objective
		t_grad = tf.gradients(t_score, t_input)[0]

		#split the image into a number of octaves
		img = img0
		octaves = []
		for _ in range(octave_n-1):
			hw = img.shape[:2]
			lo = resize(imf, np.int32(np.float32(hw)/octave_scale))
			hi = img-resize(how)
	
