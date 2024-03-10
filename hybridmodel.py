import os
import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras.utils as image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from shutil import copyfile
from flask import Flask, flash, request, redirect, url_for, render_template,send_file
w=1

#The paths to the dataset.
training_path = r"D:\projects\optimized hybrid deep grid model\techgium_ui\techgium\techgium_practise\train"
test_path = r"D:\projects\optimized hybrid deep grid model\techgium_ui\techgium\techgium_practise\test"

#Plot some images in the dataset.
def plot_image(path, int_type):
    _path = path
    label = None
    if int_type == 0:
        _path = _path + "/" + "without_grid"
        label = 0
    else:
        _path = _path + "/" + "with_grid"
        label = 1
    
    plt.figure(figsize=(30, 30))
    plt.subplots_adjust(top=None, bottom=None, left=None, right=None, wspace=0.2, hspace=0.5)
    
    lst_img_name = os.listdir(_path)
    for i in range(1, 17, 1):
        th = np.random.randint(0, len(lst_img_name) - 1)
        plt.subplot(4, 4, i)
        img = _path + "/" + lst_img_name[th]
        img = cv.imread(img)
        plt.imshow(img)
        if label == 0:
            plt.title("without")
        else:
            plt.title("with")

train_data=ImageDataGenerator(rescale=1./255,
                              rotation_range=40,
#                               width_shift_range=0.2,
#                               height_shift_range=0.2,
#                               shear_range=0.2,
#                               zoom_range=0.2,
                              horizontal_flip=0.2)

test_data=ImageDataGenerator(rescale=1./255,
                             rotation_range=40,
#                              width_shift_range=0.2,
#                              height_shift_range=0.2,
#                              shear_range=0.2,
#                              zoom_range=0.2,
                             horizontal_flip=0.2)

train_data=train_data.flow_from_directory(directory=training_path, batch_size=32, target_size=(200, 200), class_mode='binary')
test_data=test_data.flow_from_directory(directory=test_path, batch_size=32, target_size=(200, 200), class_mode='binary')



# defining the model for cnn
def my_model():
    model = tf.keras.models.Sequential([
        
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        
            tf.keras.layers.CenterCrop(180, 180),
        
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Dropout(0.2),
        
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Dropout(0.2),
        
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



import scipy
model = my_model()

#running epoch on cnn
history=model.fit_generator(train_data, epochs=2, validation_data=test_data)

from keras.preprocessing import image
def another_function(class_1_images):
  import numpy as np
  import cv2
  from keras.layers import Conv2D, Input, UpSampling2D
  from skimage.restoration import denoise_tv_chambolle
  from keras.models import Model
  from keras.optimizers import Adam

  # Load the X-ray image
  img = cv2.imread(class_1_images)

# Check the image type
  if img.dtype == np.uint16:
    img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))

# Convert the image to float32 type
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = np.float32(img)
  
  # Apply a Gaussian band-stop filter to remove gridline artifacts
  dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
  dft_shift = np.fft.fftshift(dft)
  rows, cols = img.shape
  crow, ccol = rows // 2, cols // 2
  mask = np.zeros((rows, cols, 2), np.uint8)
  mask[crow-90:crow+90, ccol-90:ccol+90] = 1
  fshift = dft_shift * mask
  f_ishift = np.fft.ifftshift(fshift)
  img_back = cv2.idft(f_ishift)
  img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

  # Optimize image using Adam optimizer
  opt = Adam(learning_rate=0.001)
  img_optimized = denoise_tv_chambolle(img_back, weight=0.1, eps=0.0002, n_iter_max=200, multichannel=False)

  # Normalize the image
  img_filt = cv2.normalize(img_optimized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

  #plotting the images
  fig, axs = plt.subplots(1, 2, figsize=(15, 15))
  axs[0].imshow(img, cmap='gray')
  axs[0].set_title("Original Image")
  axs[1].imshow(img_filt, cmap='gray')
  axs[1].set_title("Filtered Image")
  cv2.imwrite("x-ray_without_gridlines.jpg", img_filt)
  plt.show()
 

def print_fun():
    w=0
    print("GRIDLINES ARE NOT PRESENT IN THE GIVEN IMAGE. PLEASE PROVIDE THE IMAGE WITH GRIDLINE ARTIFACTS")
    
import numpy as np    

def standardize_image(image):
    if len(image.shape) == 2:
        # Grayscale image
        return np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[-1] == 4:
        # RGBA image
        return image[..., :3]
    elif len(image.shape) == 3 and image.shape[-1] == 1:
        # Single-channel image
        return np.repeat(image, 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[-1] == 2:
        # 2-channel image
        return np.concatenate([image, np.zeros_like(image[..., :1])], axis=-1)
    else:
        # RGB image or unknown format
        return image[..., :3]


def predict1(s):
    from PIL import Image
    import numpy as np
    #from keras.preprocessing.image import load_img

    path = r"D:\\projects\\optimized hybrid deep grid model\\techgium_ui\\techgium\\static\\images\\"+s
    img = Image.open(path)
    img = img.resize((200, 200))
    #img = img.resize((64, 64))
    img = standardize_image(np.array(img))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    b=0
    if result[0][0] == 1:
        prediction = 'with grid'
        b=1
    else:
        prediction = 'without grid'
    
   

    if(prediction=='with grid'):
        another_function(path)
    else:
        print_fun()
    # print("hybrid w=",w,"b=",b)
    print("hybrid",prediction)
    return b