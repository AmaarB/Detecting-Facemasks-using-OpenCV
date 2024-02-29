import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow import keras
from retinaface import RetinaFace 


# load mask-detection model
#loading a saved model from ('model_building_facemask_classifcation) file in line 80.
model = keras.models.load_model('model.h5')

# function to identify human faces and detect face masks
def inference(img):
  # change color channels of input image
  #The diffrence between BGR and RGB is the arrangement of subpixels for RED,GREEN,BLUE
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # through the use of facial detector (retinaface) we can determine the facial area co-ordinates and some landmarks (eyes, nose and mouth)
  resp = RetinaFace.detect_faces(img)
  
  # empty list to store cropped face images
  faces = []

  # extract faces using line 20 resp
  for i in range(len(resp)):
    # get coordinates of each face in the input image
    x1,y1,x2,y2 = resp[f'face_{i+1}']['facial_area']
    # add detected faces to the list
    faces.append(img[y1:y2, x1:x2, :])
  
  # empty list to store model predictions for each face
  labels = []
  
  # get predictions for face masks (0: with mask, 1: no mask)
  for f in faces:
    # resize face image before passing it to model
    f = cv2.resize(f, (20, 20))
    # normalize the pixel values of the face image between 0 and 1
    f = f.astype("float32")/255
    # get prediction from the model
    # by using the model which was trained previously we can get a prediction
    p = model.predict(f.reshape(1, f.shape[0], f.shape[1], f.shape[2]))

    # find the label with maximum probability score, either 0 or 1
    pred_idx = np.argmax(p)
    # convert probability to confidence percentage
    confidence = round(100 * p[0][pred_idx], 1)
    
    if pred_idx == 0:
      labels.append(["with mask",confidence])
    else:
      labels.append(["without mask",confidence])

  true_color_faces = []
  # recolor face images with true colors
  for f in faces:
    temp = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    true_color_faces.append(temp)
  
  # return cropped face images and their predicted labels
  return (true_color_faces, labels)

# Title of your app
st.title('COVID Face Mask Detection 😷')

# function to read the image
def load_image(x):
  img = Image.open(x)
  return img

# file upload widget
image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

if image_file is not None:
    image = Image.open(image_file)
    # display uploaded image
    st.image(image, caption='Input image')
    # convert image to array that will be passed to the model
    img_array = np.array(image)

    if st.button('Detect Face Masks'):
        # get predictions on the uploaded image
        faces, labels = inference(img_array[:,:,:3])
        
        # display cropped faces with their predicted labels and confidence score
        for f,l in zip(faces,labels):
            st.image(f)
            st.write(f"{l[0]} - {l[1]}%")
    else:
        pass