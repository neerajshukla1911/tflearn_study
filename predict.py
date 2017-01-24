from models.vgg import VGG
from tflearn import data_utils


model = VGG().create([None, 128, 128, 3], 4)
model.load('trained_models/vgg/fynd/my_model.tflearn')

img = data_utils.load_image('/home/neeraj/AI/aml_data/fynd/2.jpg')
img = data_utils.resize_image(img, 128, 128)
img = data_utils.pil_to_nparray(img)
print(model.predict_label([img]))
