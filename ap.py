from flask import Flask
from routes.face_recognition import face_recognition_bp

# Initialize Flask
app = Flask(__name__)

# Register the blueprint
app.register_blueprint(face_recognition_bp)

if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.31')
