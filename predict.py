from models.alexnet import AlexNet
from tflearn import data_utils
import h5py
import os


model = AlexNet().create([None, 128, 128, 3], 4)

model.load('trained_models/alexnet/my_model.tflearn', weights_only=True)

img = data_utils.load_image('{}/{}'.format( os.getenv("HOME"),'AI/data/2.jpg')
img = data_utils.resize_image(img, 128, 128)
img = data_utils.pil_to_nparray(img)
print(model.predict_label([img]))


#load trained model and predict labels
# model = VGG16().create([None, 224, 224, 3], 4)
# model = VGG16().create([None, 224, 224, 3], 4)
# model.load('/home/neeraj/AI/models/vgg16.tflearn', weights_only=True)
# img = data_utils.load_image('/home/neeraj/AI/aml_data/fynd/2.jpg')
# img = data_utils.resize_image(img, 224, 224)
# img = data_utils.pil_to_nparray(img)
# print(model.predict_label([img]))
# model = VGG16().create([None, 224, 224, 3], 4)
