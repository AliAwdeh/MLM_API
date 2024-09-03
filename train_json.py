import os
import numpy as np
import faiss
import pandas as pd
import pickle
import json
from utils.feature_extraction import extract_image_features
from sklearn.preprocessing import OneHotEncoder

# Directories
training_dir = 'images'
inventory_dir = 'inventory'
version_dir = 'model_versions'
json_dir = 'styles'  # Directory where the JSON files are stored

# Ensure the version directory exists
if not os.path.exists(version_dir):
    os.makedirs(version_dir)

# Load and preprocess CSV data for training
styles_df = pd.read_csv('styles.csv')
styles_df = styles_df.drop(['id', 'productDisplayName'], axis=1)

encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(styles_df[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage']]).toarray()

# Initialize FAISS index for similarity search
dimension = encoded_features.shape[1] + 2048  # Metadata + image feature dimensions
index = faiss.IndexFlatL2(dimension)
combined_features = np.empty((0, dimension))

# Function to get the next model version
def get_next_version():
    existing_versions = [int(f.replace('mlmv', '')) for f in os.listdir(version_dir) if f.startswith('mlmv')]
    next_version = max(existing_versions, default=-1) + 1
    return f"mlmv{next_version}"

# Function to load JSON metadata
def load_json_metadata(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Function to extract features from JSON data
def extract_features_from_json(json_data):
    features = {}
    features['price'] = json_data['data'].get('price', 0)
    features['discountedPrice'] = json_data['data'].get('discountedPrice', 0)
    features['brandName'] = json_data['data'].get('brandName', 'Unknown')
    features['season'] = json_data['data'].get('season', 'Unknown')
    features['year'] = json_data['data'].get('year', 'Unknown')
    features['pattern'] = json_data['data']['articleAttributes'].get('Pattern', 'Unknown')
    # Additional features can be extracted as needed
    return features

# Encode new features from JSON and combine with existing features
def encode_and_combine_features(json_data, image_features):
    # Extract features from JSON
    json_features = extract_features_from_json(json_data)

    # Create a DataFrame for the JSON features to apply OneHotEncoder
    json_df = pd.DataFrame([json_features])

    # Encode the JSON features
    json_encoded = encoder.transform(json_df).toarray()

    # Combine encoded JSON features with image features
    combined_feature = np.hstack((json_encoded, image_features))
    return combined_feature

# Train the model on the training dataset
print("Training the model on the training dataset...")

for image_name in os.listdir(training_dir):
    image_path = os.path.join(training_dir, image_name)
    features = extract_image_features(image_path)

    # Load the corresponding JSON metadata
    product_id = os.path.splitext(image_name)[0]
    json_path = os.path.join(json_dir, f'{product_id}.json')
    
    if os.path.exists(json_path):
        json_data = load_json_metadata(json_path)
        combined_feature = encode_and_combine_features(json_data, features)
        combined_features = np.vstack((combined_features, combined_feature))
    else:
        print(f"JSON metadata not found for {image_name}, skipping...")

# Add inventory images to the FAISS index
print("Indexing inventory images...")

for image_name in os.listdir(inventory_dir):
    image_path = os.path.join(inventory_dir, image_name)
    features = extract_image_features(image_path)

    # For inventory, you may or may not have JSON data
    product_id = os.path.splitext(image_name)[0]
    json_path = os.path.join(json_dir, f'{product_id}.json')
    
    if os.path.exists(json_path):
        json_data = load_json_metadata(json_path)
        combined_feature = encode_and_combine_features(json_data, features)
        combined_features = np.vstack((combined_features, combined_feature))
    else:
        print(f"JSON metadata not found for {image_name}, using only image features.")
        combined_feature = np.hstack((np.zeros(encoded_features.shape[1]), features))  # Fallback to image features only
        combined_features = np.vstack((combined_features, combined_feature))

index.add(combined_features.astype('float32'))

# Save the new model version
version = get_next_version()
version_path = os.path.join(version_dir, version)
os.makedirs(version_path)

faiss.write_index(index, os.path.join(version_path, f'{version}_index.bin'))
np.save(os.path.join(version_path, f'{version}_features.npy'), combined_features)

with open(os.path.join(version_path, f'{version}_encoder.pkl'), 'wb') as f:
    pickle.dump(encoder, f)

print(f"Model trained and saved successfully as version {version} on this device!")