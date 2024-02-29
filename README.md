# Detecting Facemasks using OpenCV, Keras/TensorFlow and Deep Learning

This code implements a Convolutional Neural Network (CNN)-based face mask detection system. Preprocessing face mask image data is the first step, after which faces are cropped and saved using labelled annotations. TensorFlow and Keras are used to construct the CNN model, which is trained on preprocessed data. Accuracy and loss graphs are used to illustrate the training process. After that, users can upload photos for real-time face mask detection using the model that has been loaded into the Streamlit web app. In order to determine if a person is wearing a mask or not, the software extracts faces using RetinaFace, runs them through the trained model, and makes the prediction. The identified faces are shown with their labels and confidence ratings.

<img width="1019" alt="Screenshot 2024-02-29 at 20 41 53" src="https://github.com/AmaarB/Detecting-Facemasks-using-OpenCV/assets/84424799/760fdf72-8c6d-4b20-9b76-32fb283ea1c0">
<img width="500" alt="Screenshot 2024-02-29 at 20 42 16" src="https://github.com/AmaarB/Detecting-Facemasks-using-OpenCV/assets/84424799/c6822dd8-b8e1-467c-885a-cd59677e4f51">

The model's success in accurately guessing whether a face is wearing a mask or not during training is represented by accuracy, which is graphed in the code. Better model performance is indicated by higher accuracy numbers. The accuracy graph provides a visual representation of the model's learning over several epochs from the training data.
