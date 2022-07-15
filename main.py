import streamlit as st
import os
import pickle
from PIL import Image
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

st.title("Fashion Recommender System")

feature_list=np.array(pickle.load(open('images_feature.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable =False


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable =False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('upload',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
            st.header("File uploaded successfully")
        return 1
    except:
        return 0

def extractFeatures(img_path,model):
    img=image.load_img(img_path,target_size=(224,224,3))
    img = image.img_to_array(img)
    expanded_img = np.expand_dims(img, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result =model.predict(preprocessed_img).flatten()
    normalised_result = result/norm(result)

    return normalised_result

def recommend_cloth(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric='euclidean')
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([features])
    return indices[0]

def show_recommendation(indices):
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(filenames[indices[0]])

    with col2:
        st.image(filenames[indices[1]])

    with col3:
        st.image(filenames[indices[2]])

    with col4:
        st.image(filenames[indices[3]])

    with col5:
        st.image(filenames[indices[4]])

uploaded_file= st.file_uploader('Choose an image')
if uploaded_file is not None:
    # st.text("Image uploaded")
    if save_uploaded_file(uploaded_file):
        # st.text("Image saved")
        display_image = Image.open(uploaded_file)
        st.image(display_image,width=None)
        # pass
        features = extractFeatures(os.path.join('upload',uploaded_file.name),model)
        indices = recommend_cloth(features,feature_list)
        show_recommendation(indices)


    else:
        st.header("Some error occured! Please try again")
