import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'thirdpage.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class SecondPage extends StatefulWidget {
  const SecondPage({Key? key}) : super(key: key);

  @override
  _SecondPageState createState() => _SecondPageState();
}

class _SecondPageState extends State<SecondPage> {
  File? _selectedImage;
  bool _isTranslating = false;  // Added state variable

  // take image from camera
  Future<void> _getImageFromCamera() async {
    final XFile? image =
        await ImagePicker().pickImage(source: ImageSource.camera);
    _handleImageSelection(image);
  }

  // select image from gallery
  Future<void> _getImageFromGallery() async {
    final XFile? image =
        await ImagePicker().pickImage(source: ImageSource.gallery);
    _handleImageSelection(image);
  }

  void _handleImageSelection(XFile? image) {
    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
      });
    }
  }

  void _clearSelectedImage() {
    setState(() {
      _selectedImage = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: <Widget>[
          // Background Image
          Image.asset(
            'assets/background.jpg',
            width: double.infinity,
            height: double.infinity,
            fit: BoxFit.cover,
          ),
          // Background Container with Opacity
          Container(
            color: const Color.fromARGB(255, 90, 80, 80).withOpacity(0.7),
            width: double.infinity,
            height: double.infinity,
          ),
          Center(
            child: SingleChildScrollView(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: <Widget>[
                  const SizedBox(height: 50), // Add some space at the top

                  const SizedBox(height: 20),
                  // Display selected or default image
                  Container(
                    width: 350,
                    height: 300,
                    decoration: BoxDecoration(
                      border: Border.all(
                          color: Colors.white,
                          width: 2
                      ),
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: _selectedImage != null
                        ? Image.file(
                      _selectedImage!,
                      width: 250,
                      height: 250,
                      fit: BoxFit.cover,
                    )
                        : Image.asset(
                      'assets/addimage.png',
                      width: 250,
                      height: 250,
                      fit: BoxFit.cover,
                    ),
                  ),
                  const SizedBox(height: 20),
                  // Buttons for capturing and selecting images
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      IconButton(
                        onPressed: _getImageFromCamera,
                        icon: const Icon(Icons.camera),
                        iconSize: 50,
                        color: Colors.white,
                      ),
                      const SizedBox(width: 50),
                      IconButton(
                        onPressed: _getImageFromGallery,
                        icon: const Icon(Icons.image),
                        iconSize: 50,
                        color: Colors.white,
                      ),
                    ],
                  ),
                  const SizedBox(height: 40),
                  // Button for prediction
                  ElevatedButton(
                    onPressed: () async {
                      if (_selectedImage != null) {
                        setState(() {
                          _isTranslating = true;  // Set to true when translating starts
                        });

                        var request = http.MultipartRequest(
                            'POST',
                            Uri.parse(
                                'http://192.168.43.76:5000/upload_and_predict'));

                        request.files.add(await http.MultipartFile.fromPath(
                            'image', _selectedImage!.path));

                        try {
                          var response = await request.send();

                          if (response.statusCode == 200) {
                            var responseBody = await response.stream.bytesToString();
                            var result = jsonDecode(responseBody);
                            var predictedClass = result['predicted_class'];
                            print('Predicted Class: $predictedClass');

                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => ThirdPage(
                                  selectedImage: _selectedImage!,
                                  predictedClass: predictedClass,
                                ),
                              ),
                            ).then((_) {  // Set to false when navigation is complete
                              setState(() {
                                _isTranslating = false;
                              });
                            });

                          } else {
                            // Handle other status codes
                            print('Error sending image: ${response.statusCode}');
                            setState(() {
                              _isTranslating = false;
                            });
                          }
                        } catch (e) {
                          print('Error sending image: $e');
                          setState(() {
                            _isTranslating = false;
                          });
                        }

                      } else {
                        // Handle case when no image is selected
                        print('No image selected');
                      }
                    },
                    style: ElevatedButton.styleFrom(
                      foregroundColor: Colors.white,
                      backgroundColor: Colors.blue,
                      padding: const EdgeInsets.all(15),
                    ),
                    child: _isTranslating  // Check if translating
                        ? CircularProgressIndicator(  // Display loading indicator
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    )
                        : const Text(
                      'Translate',
                      style: TextStyle(fontSize: 18),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
