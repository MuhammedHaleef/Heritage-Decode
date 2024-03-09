from flask import Flask, jsonify, request

app = Flask(__name__)

# Define a simple route
@app.route('/api/data', methods=['GET'])
def get_data():
    data = {'message': 'Hello from Python backend!'}
    return jsonify(data)

# Define a route to handle POST requests
@app.route('/api/post_data', methods=['POST'])
def post_data():
    request_data = request.get_json()
    received_data = request_data.get('data')
    # Process the received data here
    response = {'message': 'Data received successfully', 'data': received_data}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
