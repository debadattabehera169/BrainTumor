import cv2
import os
from PIL   import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import to_categorical # for categorical_cross entropy

image_directory='Datasets/'

no_tumor_iamges=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory +'yes/')
dataset=[]
label=[]

input_size=64

#print(no_tumor_iamges)
#path='no0.jpg'
#print(path.split('.')[1])

for i,image_name in enumerate(no_tumor_iamges):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+ 'no/'+image_name)
        image=Image.fromarray(image,'RGB') #   image converted to array
        image=image.resize((input_size,input_size))# resize the all image in to same size
        dataset.append(np.array(image))
        label.append(0)

for i,image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+ 'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((input_size,input_size))
        dataset.append(np.array(image))# all the datasets
        label.append(1) # for op classification


#print(dataset)
#print(label)   
#print(len(label))  
#print(len(dataset))

dataset=np.array(dataset)
label=np.array(label)

x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=0)


#Reshape=(n,image_width,image_height,n_channel) 

#print(x_train.shape)
#print(y_train.shape)
#print(x_train.shape)
#print(x_test.shape)

#normalize the data
x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)


#y_train=to_categorical(y_train,num_classes=2) # for categorical_cross entropy
#y_test=to_categorical(y_test,num_classes=2) # for categorical_cross entropy

#built model

model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(input_size,input_size,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3,),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
#mode.add(Dense(2)) # for categorical_cross entropy
model.add(Activation('sigmoid'))
#model.add(Activation('softmax')) # for categorical_cross entropy

#Binary CrossEntropy =1, sigmoid
#categorical CrossEntropy=2 ,softmax

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer=='adam',metrics=['accuracy']) # for categorical_cross entropy

model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=5,validation_data=(x_test,y_test),shuffle=False)
model.save('BrainTumor5epochs.h5')




