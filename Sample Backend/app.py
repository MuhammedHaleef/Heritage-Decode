from flask import Flask, request, jsonify
from PIL import Image
from collections import Counter
import io

app = Flask(__name__)


def get_major_color(image):
    # Open the image
    image = Image.open(io.BytesIO(image))

    # Convert the image to RGB mode if it's not already in that mode
    image = image.convert('RGB')

    # Get the colors from the image
    colors = list(image.getdata())

    # Count the occurrence of each color
    color_counts = Counter(colors)

    # Find the color with the highest count
    major_color = max(color_counts, key=color_counts.get)

    # Return the major color as a string
    if major_color[0] > major_color[1] and major_color[0] > major_color[2]:
        return "red"
    elif major_color[1] > major_color[0] and major_color[1] > major_color[2]:
        return "green"
    else:
        return "blue"  # Default to blue if no other condition is met


@app.route('/', methods=['POST'])
def index():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_data = file.read()
    major_color = get_major_color(image_data)

    return jsonify({'major_color': major_color})


if __name__ == '__main__':
    app.run(debug=True)
