import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable =False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])


feature_list=np.array(pickle.load(open('images_feature.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))

img = image.load_img('sample/jersey.jpg', target_size=(224, 224, 3))
img = image.img_to_array(img)
expanded_img = np.expand_dims(img, axis=0)
preprocessed_img = preprocess_input(expanded_img)
result = model.predict(preprocessed_img).flatten()
normalised_result = result / norm(result)



neighbors = NearestNeighbors(n_neighbors=5,algorithm = "brute",metric='euclidean')
neighbors.fit(feature_list)

distance,indices=neighbors.kneighbors([normalised_result])

for file in indices[0]:
    print(filenames[file])