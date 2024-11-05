import cv2
import os
from mtcnn import MTCNN

# Folder input dan output
input_folder = 'databaru/Bagus'  # Ganti dengan path folder gambar input
output_folder = 'databaru/BagCropped'  # Ganti dengan path folder output

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Inisialisasi MTCNN
detector = MTCNN()

# Loop melalui semua file dalam folder input
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Cek ekstensi file
        # Load gambar
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Deteksi wajah
        faces = detector.detect_faces(image)

        for face in faces:
            x, y, width, height = face['box']
            # Tambahkan margin
            margin = 20  # Ubah sesuai kebutuhan
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            width = width + margin * 2
            height = height + margin * 2

            # Crop dan resize
            cropped_face = image[y:y+height, x:x+width]
            cropped_face_resized = cv2.resize(cropped_face, (255, 255))

            # Simpan gambar cropped
            output_path = os.path.join(output_folder, f'cropped_{filename}')
            cv2.imwrite(output_path, cropped_face_resized)

print("Proses cropping selesai!")
