from keras.models import load_model


def get_model(model_path):
	model = None
	try:
		# create log directory for TensorBoard
		# if not os.path.isdir(model_name+"/logs"):
		# 	os.makedirs(model_name+"/logs")

		# load the pre-trained network
		print("[INFO] loading pre-trained network...")
		model = load_model(model_path)

	except:
		print("[INFO] pre-trained network model '{}' not found".format(model_path))
	
	return model


def save_model(model_name, model):
	# create log directory for TensorBoard
	# if not os.path.isdir(model_name+"/logs"):
	# 	os.makedirs(model_name+"/logs")
	
	# save the network to disk
	print("[INFO] serializing network...")
	mname = "{}.hdf5".format(model_name)
	model.save(mname)
