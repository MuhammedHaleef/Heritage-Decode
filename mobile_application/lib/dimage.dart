import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<void> sendPhoto(File photo) async {
  // Encode the photo as base64
  String base64Photo = base64Encode(photo.readAsBytesSync());

  // Send an HTTP request to the Flask back-end
  String url = 'http://localhost:8000/api/upload';
  Uri uri = Uri.parse(url); // Convert string URL to Uri object
  Map<String, String> headers = {'Content-Type': 'application/json'};
  Map<String, dynamic> body = {'photo': base64Photo, 'option': 'Deuteranophia'};
  http.Response response =
      await http.post(uri, headers: headers, body: jsonEncode(body));

  // Handle the response from the Flask back-end
  if (response.statusCode == 200) {
    print('Response: ${response.body}');
  } else {
    print('Error: ${response.body}');
  }
}

void main() async {
  // Replace 'path_to_photo' with the actual path to your photo file
  File photo = File('path_to_photo');

  // Send the photo
  await sendPhoto(photo);
}
