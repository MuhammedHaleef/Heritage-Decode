from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    # Receive image data from Flutter app
    image_data = request.files['image'].read()

    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the image to a 2D array
    pixels = image_rgb.reshape(-1, 3)

    # Convert pixel values to a list of tuples (R, G, B)
    pixel_list = [tuple(pixel) for pixel in pixels]

    # Get unique colors and their counts
    unique_colors, color_counts = np.unique(pixel_list, axis=0, return_counts=True)

    # Sort colors by counts in descending order
    sorted_indices = np.argsort(-color_counts)
    sorted_colors = unique_colors[sorted_indices]
    sorted_counts = color_counts[sorted_indices]

    # Prepare response
    response_data = []
    for i in range(min(10, len(sorted_colors))):
        color = sorted_colors[i]
        count = sorted_counts[i]
        response_data.append({
            "color": [int(c) for c in color],
            "count": int(count)
        })

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)  # Run Flask app in debug mode for development
