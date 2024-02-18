from flask import Flask, request, jsonify
from PIL import Image
from collections import Counter
import io

app = Flask(__name__)

def get_image_main_color(image):
    # Open the image
    image = Image.open(io.BytesIO(image))

    # Convert the image to RGB mode if it's not already in that mode
    image = image.convert('RGB')

    # Get the colors from the image
    colors = list(image.getdata())

    # Count the occurrence of each color
    color_counts = Counter(colors)

    # Find the color with the highest count
    main_color = max(color_counts, key=color_counts.get)

    # Return the main color as a string
    if main_color[0] > main_color[1] and main_color[0] > main_color[2]:
        return "red"
    elif main_color[1] > main_color[0] and main_color[1] > main_color[2]:
        return "green"
    else:
        return "blue"  # Default to blue if no other condition is met

@app.route('/get_main_color', methods=['POST'])
def get_main_color():
    # Receive image data from request
    image_data = request.files['image'].read()

    # Get main color
    main_color = get_image_main_color(image_data)

    # Return main color as JSON response
    return jsonify({"main_color": main_color})

if __name__ == "__main__":
    app.run(debug=True)
