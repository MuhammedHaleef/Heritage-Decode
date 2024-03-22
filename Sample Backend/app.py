from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

@app.route("/")
def index():
    return "testing heritage decode backend server"
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.form:
        return jsonify({'error': 'No image data found'})

    base64_image = request.form['image']
    image_bytes = base64.b64decode(base64_image)

    with open('uploaded_image.jpg', 'wb') as f:
        f.write(image_bytes)

    return jsonify({'message': 'Image uploaded successfully'})


if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Change the port to 8000 or any other available port

