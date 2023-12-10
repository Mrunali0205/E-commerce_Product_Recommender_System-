# ResNet50 model pretrained on ImageNet and removes the top classification layer. It adds a GlobalMaxPooling2D layer to obtain a fixed-size vector representation for each image.
# Defined a function named extract_features which will extract features from all training data set and store the results in variable normalized_result.
# Stored path of all the images in variable "filenames" with the help of os library.
# Stored all the features of training data in variable feature_list
# Created two folders named embeddings and fimenames in project to store the extracted features and filenames respectively. Used pickle library to do this.

# Stored the features and paths in two variables named feature_list and filenames.
# Used same ResNet50 model to extract features of new test image.
# Used K-NearestNeighbors to predict 5 closest images. Used euclidean distance.

# Used streamlit library to create a website.
# Created a folder named "uploads" in the project to store the new test image. Also displayed the image.
# Then used the features and paths stored in feature_list and filenames to predict the closest 5 images using KNN.
# Displayed those 5 images as well.


import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# Set page config to wide mode
st.set_page_config(layout="wide")

# Custom CSS for a stylish background and other styles
st.markdown(
    """
    <style>
    body {
        background: url('your_background_image_url') no-repeat center center fixed;
        background-size: cover;
        background-color: #1E90FF;  /* Fallback color if the image doesn't load */
        color: #FFFFFF;
    }
    .big-font {
        font-size: 36px !important;
        color: #FF6347 !important;
    }
    .highlight {
        background-color: #FFD700;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("👗 Fashion Recommender System")
st.markdown("<p class='big-font'>Upload an image and discover your style!</p>", unsafe_allow_html=True)

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Rest of your code remains unchanged


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result/norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbours.fit(feature_list)

    distances, indices = neighbours.kneighbors([features])
    return indices


# Steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # Feature Extraction
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        st.text(features)
        # recommendation
        indices = recommend(features,feature_list)
        # Show
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])


    else:
        st.header("Some error occured in file upload")


