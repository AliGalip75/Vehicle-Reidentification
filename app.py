from flask import Flask, request, jsonify, render_template
from inference import similarity_of_two_images
import os

app = Flask(__name__)

UPLOAD_FOLDER = "tmp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    img1 = request.files['image1']
    img2 = request.files['image2']

    # Geçici dosya yolları
    path1 = os.path.join(UPLOAD_FOLDER, img1.filename)
    path2 = os.path.join(UPLOAD_FOLDER, img2.filename)
    img1.save(path1)
    img2.save(path2)

    # Benzerlik hesapla
    cosine = similarity_of_two_images(path1, path2)
    similarity_percent = ((cosine + 1) / 2) * 100  # 0–100 arası yüzdelik form
    result = {
        "cosine": cosine,
        "similarity_percent": similarity_percent
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
