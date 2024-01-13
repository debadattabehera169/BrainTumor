import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor5epochs.h5')

image=cv2.imread('E:\B_Datasets\pred\pred45.jpg')

img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)

input_img=np.expand_dims(img,axis=0)


#print(img)

result=model.predict_classes(input_img)
print(result)