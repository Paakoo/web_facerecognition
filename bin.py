from flask import Flask, render_template, request, redirect, url_for, send_file, Response, jsonify
from werkzeug.utils import secure_filename
from deepface import DeepFace
import os
import pandas as pd
import base64
from datetime import datetime
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from retinaface import RetinaFace
import pickle

# Initialize Flask
app = Flask(__name__)

# Configure upload folder and allowed extensions
BASE_FOLDER = 'databaru/'
UPLOAD_FOLDER = 'static/uploads/'
CROPPED_FOLDER = 'static/cropped/'  # Folder to save cropped images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CROPPED_FOLDER'] = CROPPED_FOLDER
 
# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load existing embeddings from .pkl file
# Fungsi untuk memuat embedding dari file pkl yang sudah ada
def load_existing_embeddings(pkl_file):
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as file:
            existing_embeddings = pickle.load(file)
        return existing_embeddings
    return {}

# Fungsi untuk menyimpan embedding ke file pkl
def save_embeddings_to_pkl(embeddings, pkl_file):
    with open(pkl_file, 'wb') as file:
        pickle.dump(embeddings, file)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Fungsi untuk menyimpan gambar
def save_image(image_data, folder, angle, count):
    # Decode base64 image data
    image_data = image_data.split(",")[1]
    image_data = base64.b64decode(image_data)

    # Buat nama file berdasarkan waktu dan angle
    filename = f"{angle}_{count}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
    file_path = os.path.join(folder, filename)

    # Simpan gambar ke file
    with open(file_path, 'wb') as f:
        f.write(image_data)

    return filename

@app.route('/capture')
def capture():
    return render_template('captureWajah.html')

@app.route('/save_image', methods=['POST'])
def save_image_route():
    data = request.get_json()
    angle = data.get('angle')
    count = data.get('count')
    image_data = data.get('image')
    username = data.get('username')

    # Buat folder pengguna jika belum ada
    user_folder = os.path.join(BASE_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)

    try:
        filename = save_image(image_data, user_folder, angle, count)
        return jsonify({'message': f'Image saved as {filename}'})
    except Exception as e:
        print(f"Error saving image: {e}")
        return jsonify({'message': 'Error saving image'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # Menyimpan file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # File pkl untuk menyimpan embedding
        pkl_file = 'ds_model_facenet512_detector_retinaface_aligned_normalization_base_expand_0.pkl'

        # Muat embedding yang sudah ada
        existing_embeddings = load_existing_embeddings(pkl_file)

        if file_path not in existing_embeddings:
            try:
                # Menggunakan DeepFace untuk menemukan identitas
                res = DeepFace.find(img_path=file_path, db_path=BASE_FOLDER, 
                                    model_name='Facenet512', detector_backend='retinaface', 
                                    enforce_detection=True, distance_metric='cosine')

                print(f"Result from DeepFace.find(): {res}")
                
                if len(res) > 0:
                    identity = res[0]['identity'][0]
                    name = os.path.basename(os.path.dirname(identity))
                    result = {'image': filename, 'name': name}

                    # Tambahkan embedding baru ke dictionary existing_embeddings
                    existing_embeddings[file_path] = res[0]

                    # Simpan kembali embedding baru ke file pkl
                    save_embeddings_to_pkl(existing_embeddings, pkl_file)
                else:
                    result = {'image': filename, 'name': "Unknown"}

            except Exception as e:
                print(f"Error dalam memproses {filename}: {e}")
                result = {'image': filename, 'name': "Unknown"}
        else:
            print(f"Embedding untuk {file_path} sudah ada")
            result = {'image': filename, 'name': "Already Exists"}

        return render_template('result.html', result=result)
    else:
        return redirect(request.url)


@app.route('/upload_camera', methods=['POST'])
def upload_camera():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # Menyimpan file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Melakukan face recognition
        try:
            # Menggunakan DeepFace untuk menemukan identitas
            res = DeepFace.find(img_path=file_path, db_path= BASE_FOLDER, 
                                model_name='Facenet512', detector_backend='retinaface', 
                                enforce_detection=False, distance_metric='cosine')
            
            print(f"Result from DeepFace.find(): {res}")

            if len(res) > 0:
                identity = res[0]['identity'][0]
                name = os.path.basename(os.path.dirname(identity))
                result = {'image': filename, 'name': name}
            else:
                result = {'image': filename, 'name': "Unknown"}

        except Exception as e:
            print(f"Error dalam memproses {filename}: {e}")
            result = {'image': filename, 'name': "Unknown"}
        
        # Mengembalikan hasil sebagai HTML
        return render_template('result.html', result=result)
    else:
        return redirect(request.url)


# @app.route('/upload_camera', methods=['POST'])
# def upload_camera():
#     if 'file' not in request.files:
#         return redirect(request.url)
    
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
    
#     if file and allowed_file(file.filename):
#         # Save file
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         # Perform anti-spoofing detection
#         try:
#             img = cv2.imread(file_path)
#             faces, _ = model.detect(img, threshold=0.5, max_num=1)

#             # Assuming first face found is the one to check for spoofing
#             if len(faces) > 0:
#                 face = faces[0]
#                 is_spoof = model.predict(img, face)
#                 spoofing_message = "Spoof detected!" if is_spoof else "Image is real!"
#             else:
#                 spoofing_message = "No face detected."

#             result = {
#                 'image': filename,
#                 'spoofing_message': spoofing_message
#             }

#         except Exception as e:
#             print(f"Error processing {filename}: {e}")
#             result = {
#                 'image': filename,
#                 'spoofing_message': f"Error: {str(e)}"
#             }

#         # Return result as HTML
#         return render_template('resultCamera.html', result=result)
#     else:
#         return redirect(request.url)


# Route to display result
@app.route('/result')
def result():
    return render_template('result.html')

# Route for camera page
@app.route('/camera')
def camera():
    return render_template('kamera.html')

# Running the Flask application
if __name__ == '__main__':
    app.run(debug=True)


# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         # Menyimpan file
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
        
#         # Load existing embeddings from pkl file
#         pkl_path = 'databaru/embeddings.pkl'
#         embeddings = load_embeddings(pkl_path)
        
#         # Check if the image has been processed before
#         if filename in embeddings:
#             name = embeddings[filename]['name']
#             result = {'image': filename, 'name': name}
#         else:
#             try:
#                 # Melakukan face recognition pada gambar baru
#                 res = DeepFace.find(img_path=file_path, db_path='databaru/', 
#                                     model_name='Facenet512', detector_backend='retinaface', 
#                                     enforce_detection=True, distance_metric='cosine')

#                 # Debugging: Print the structure of `res`
#                 print(f"Result from DeepFace.find(): {res}")
                
#                 # Check if res is a DataFrame or list of DataFrames
#                 if isinstance(res, list) and len(res) > 0 and isinstance(res[0], pd.DataFrame):
#                     # Access the first row of the first DataFrame
#                     if not res[0].empty and 'identity' in res[0].columns:
#                         identity = res[0].iloc[0]['identity']  # Access identity in the first row
#                         name = os.path.basename(os.path.dirname(identity))
#                         result = {'image': filename, 'name': name}

#                         # Save new embedding to the embeddings dictionary
#                         embeddings[filename] = {'name': name}
#                         save_embeddings(pkl_path, embeddings)  # Save updated embeddings to .pkl
#                     else:
#                         result = {'image': filename, 'name': "Unknown"}
#                 else:
#                     result = {'image': filename, 'name': "Unknown"}
#             except Exception as e:
#                 print(f"Error dalam memproses {filename}: {e}")
#                 result = {'image': filename, 'name': "Unknown"}
        
#         return render_template('result.html', result=result)
#     else:
#         return redirect(request.url)



    
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         # Menyimpan file
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#         # Melakukan face recognition
#         try:
#             # Menggunakan DeepFace untuk menemukan identitas
#             res = DeepFace.find(img_path=file_path, db_path='databaru/', 
#                                 model_name='Facenet512', detector_backend='retinaface', 
#                                 enforce_detection=True, distance_metric='cosine')
#             if len(res) > 0:
#                 identity = res[0]['identity'][0]
#                 name = os.path.basename(os.path.dirname(identity))
#                 result = {'image': filename, 'name': name}
#             else:
#                 result = {'image': filename, 'name': "Unknown"}

#         except Exception as e:
#             print(f"Error dalam memproses {filename}: {e}")
#             result = {'image': filename, 'name': "Unknown"}
#         return render_template('result.html', result=result)
#     else:
#         return redirect(request.url)