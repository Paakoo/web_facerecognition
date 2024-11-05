from flask import Blueprint, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename
import os
from deepface import DeepFace
from model.embedding import allowed_file, load_existing_embeddings, save_embeddings_to_pkl, save_image

# Blueprint Initialization
face_recognition_bp = Blueprint('face_recognition_bp', __name__)

# Import configurations
from config import UPLOAD_FOLDER, BASE_FOLDER

# Route for the home page
@face_recognition_bp.route('/')
def index():
    return render_template('index.html')

@face_recognition_bp.route('/camera')
def camera():
    return render_template('kamera.html')

@face_recognition_bp.route('/capture')
def capture():
    return render_template('captureWajah.html')

@face_recognition_bp.route('/save_image', methods=['POST'])
def save_image_route():
    data = request.get_json()
    angle = data.get('angle')
    count = data.get('count')
    image_data = data.get('image')
    username = data.get('username')

    user_folder = os.path.join(BASE_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)

    try:
        filename = save_image(image_data, user_folder, username, angle, count)
        return jsonify({'message': f'Image saved as {filename}'})
    except Exception as e:
        print(f"Error saving image: {e}")
        return jsonify({'message': 'Error saving image'}), 500

@face_recognition_bp.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        pkl_file = 'ds_model_facenet512_detector_retinaface_aligned_normalization_base_expand_0.pkl'
        existing_embeddings = load_existing_embeddings(pkl_file)

        if file_path not in existing_embeddings:
            try:
                res = DeepFace.find(img_path=file_path, db_path=BASE_FOLDER, 
                                    model_name='Facenet512', detector_backend='retinaface', 
                                    enforce_detection=True, distance_metric='cosine')

                print(f"Result from DeepFace.find(): {res}")
                
                if len(res) > 0:
                    identity = res[0]['identity'][0]
                    name = os.path.basename(os.path.dirname(identity))
                    result = {'image': filename, 'name': name}
                    existing_embeddings[file_path] = res[0]
                    save_embeddings_to_pkl(existing_embeddings, pkl_file)
                else:
                    result = {'image': filename, 'name': "Unknown"}

            except Exception as e:
                print(f"Error dalam memproses {filename}: {e}")
                result = {'image': filename, 'name': "Unknown"}
        else:
            result = {'image': filename, 'name': "Already Exists"}

        return render_template('result.html', result=result)
    else:
        return redirect(request.url)

@face_recognition_bp.route('/upload_camera', methods=['POST'])
def upload_camera():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            res = DeepFace.find(img_path=file_path, db_path=BASE_FOLDER, 
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
        
        return render_template('result.html', result=result)
    else:
        return redirect(request.url)
