import numpy as np
from PIL import Image
try:
    import tflite_runtime.interpreter as tflite
except:
    import tensorflow.lite as tflite

# load model
interpreter = tflite.Interpreter(model_path='model/multi_person_mobilenet_v1_075_float.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input_details:", input_details)
print("output_details:", output_details)

# input - N x H x W x C, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
img = Image.open('images/pose1.jpg').resize((width, height))
input_data = np.expand_dims(img, axis=0)
input_data = (np.float32(input_data) - 127.5) / 127.5
interpreter.set_tensor(input_details[0]['index'], input_data)
 
# run
interpreter.invoke()

# get output
print("\nOUTPUTS:")
for output_detail in output_details:
	output_data = interpreter.get_tensor(output_detail['index'])
	print("\t", output_detail["name"], "\t", output_data.shape)
