from keras.models import load_model, model_from_json
import json


with open('optimized_airfoil_nn/optimized_airfoil_nn_model.json', 'r') as json_file:
	model_json = json.load(json_file)

model = model_from_json(model_json)
model.load_weights('optimized_airfoil_nn/optimized_airfoil_nn_model.h5')

model.summary()
