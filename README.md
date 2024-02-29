# Detecting Facemasks using OpenCV, Keras/TensorFlow and Deep Learning

This code implements a Convolutional Neural Network (CNN)-based face mask detection system. Preprocessing face mask image data is the first step, after which faces are cropped and saved using labelled annotations. TensorFlow and Keras are used to construct the CNN model, which is trained on preprocessed data. Accuracy and loss graphs are used to illustrate the training process. After that, users can upload photos for real-time face mask detection using the model that has been loaded into the Streamlit web app. In order to determine if a person is wearing a mask or not, the software extracts faces using RetinaFace, runs them through the trained model, and makes the prediction. The identified faces are shown with their labels and confidence ratings.

