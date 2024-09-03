from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
import numpy as np

# Initialize the ResNet50 model pre-trained on ImageNet
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

def extract_image_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()
