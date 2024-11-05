import os

# Configuration for upload folders and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
CROPPED_FOLDER = 'static/cropped/'
BASE_FOLDER = 'databaru/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload and cropped folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)
