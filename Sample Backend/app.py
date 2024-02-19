from flask import Flask, request, render_template
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return 'No selected file'

        # If the file is selected and is an image
        if file and allowed_file(file.filename):
            # Read the image file and convert it to base64
            image_base64 = base64.b64encode(file.read()).decode("utf-8")
            # Construct HTML response with embedded image
            html_content = f"<img src='data:image/jpeg;base64,{image_base64}' style='max-width: 100%;'>"
            return render_template('index.html', image=html_content)

    return render_template('index.html', image=None)

def allowed_file(filename):
    # Add any additional checks for allowed file types here
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
