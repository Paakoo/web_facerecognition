import os
import pickle
import base64
from datetime import datetime

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Function to load existing embeddings from .pkl file
def load_existing_embeddings(pkl_file):
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as file:
            existing_embeddings = pickle.load(file)
        return existing_embeddings
    return {}

# Function to save embeddings to .pkl file
def save_embeddings_to_pkl(embeddings, pkl_file):
    with open(pkl_file, 'wb') as file:
        pickle.dump(embeddings, file)

# Function to save image data
def save_image(image_data, folder, username, angle, count):
    image_data = image_data.split(",")[1]
    image_data = base64.b64decode(image_data)

    filename = f"{username}_{angle}_{count}_{datetime.now().strftime('%Y%m%d')}.jpg"
    file_path = os.path.join(folder, filename)

    with open(file_path, 'wb') as f:
        f.write(image_data)

    return filename
