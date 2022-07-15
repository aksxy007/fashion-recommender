import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable =False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

# print(model.summary())

def extractFeatures(img_path,model):
    img=image.load_img(img_path,target_size=(224,224,3))
    img = image.img_to_array(img)
    expanded_img = np.expand_dims(img, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result =model.predict(preprocessed_img).flatten()
    normalised_result = result/norm(result)

    return normalised_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extractFeatures(file,model))

pickle.dump(feature_list,open('images_feature.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))
