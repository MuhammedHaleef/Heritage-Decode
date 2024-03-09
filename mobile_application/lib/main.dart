import 'package:flutter/material.dart';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        primaryColor: Colors.blueGrey, // Set primary color
        hintColor: Colors.orangeAccent, // Set accent color
        fontFamily: 'Roboto', // Set default font family
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Heritage Decode',
          style: TextStyle(color: Colors.white),
        ),
        leading: IconButton(
          icon: Icon(Icons.menu),
          onPressed: () {},
        ),
        backgroundColor: Theme.of(context).primaryColor, // Use primary color
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Center(
            child: Image.asset(
              'assets/Logo.png',
              height: 200.0, // Adjust image size
              width: 200.0,
            ),
          ),
          SizedBox(height: 20.0),
          ElevatedButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => SecondPage()),
              );
            },
            child: Text(
              'Get Start',
              style: TextStyle(fontSize: 20.0),
            ),
            style: ElevatedButton.styleFrom(
              padding: EdgeInsets.symmetric(horizontal: 20.0, vertical: 15.0),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10.0),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class SecondPage extends StatefulWidget {
  @override
  _SecondPageState createState() => _SecondPageState();
}

class _SecondPageState extends State<SecondPage> {
  File? _selectedImage;
  bool _isProcessing = false;

  Future<void> _getImageFromCamera() async {
    final picker = ImagePicker();
    final pickedFile = await picker.getImage(source: ImageSource.camera);

    if (pickedFile != null) {
      setState(() {
        _isProcessing = true;
      });

      // Convert image to base64
      List<int> imageBytes = await pickedFile.readAsBytes();
      String base64Image = base64Encode(imageBytes);

      // Send base64 image to backend
      await _sendImageToBackend(base64Image);

      setState(() {
        _selectedImage = File(pickedFile.path);
        _isProcessing = false;
      });
    }
  }

  Future<void> _getImageFromFileChooser() async {
    final picker = ImagePicker();
    final pickedFile = await picker.getImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _isProcessing = true;
      });

      // Convert image to base64
      List<int> imageBytes = await pickedFile.readAsBytes();
      String base64Image = base64Encode(imageBytes);

      // Send base64 image to backend
      await _sendImageToBackend(base64Image);

      setState(() {
        _selectedImage = File(pickedFile.path);
        _isProcessing = false;
      });
    }
  }

  Future<void> _sendImageToBackend(String base64Image) async {
    // Define your backend URL
    String url = 'http://127.0.0.1:8000/upload';

    // Send the base64 image to the backend
    final response = await http.post(
      Uri.parse(url),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'image': base64Image}),
    );

    if (response.statusCode == 200) {
      // Image uploaded successfully
      print('Image uploaded successfully');
    } else {
      // Error uploading image
      print('Error uploading image: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Heritage Decode',
          style: TextStyle(color: Colors.white),
        ),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Display message to select or capture image if no image is selected
          if (_selectedImage == null)
            Container(
              padding: EdgeInsets.all(20.0),
              decoration: BoxDecoration(
                color: Colors.blueGrey.withOpacity(0.2),
                borderRadius: BorderRadius.circular(10.0),
              ),
              child: Column(
                children: [
                  Text(
                    'Please select or capture an image to translate',
                    style: TextStyle(fontSize: 20.0),
                    textAlign: TextAlign.center,
                  ),
                  SizedBox(height: 10.0),
                  Icon(Icons.camera_alt, size: 50.0, color: Colors.blueGrey.withOpacity(0.6)),
                ],
              ),
            ),
          if (_selectedImage != null)
            Center(
              child: Image.file(
                _selectedImage!,
                height: 300.0,
                width: 300.0,
              ),
            ),
          SizedBox(height: 20.0),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              IconButton(
                icon: Icon(Icons.camera),
                onPressed: _isProcessing ? null : () async {
                  await _getImageFromCamera();
                },
              ),
              IconButton(
                icon: Icon(Icons.file_upload),
                onPressed: _isProcessing ? null : () async {
                  await _getImageFromFileChooser();
                },
              ),
            ],
          ),
          SizedBox(height: 20.0),
          if (_selectedImage != null)
            ElevatedButton(
              onPressed: _isProcessing ? null : () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => ThirdPage(image: _selectedImage!),
                  ),
                );
              },
              child: Text('Convert'),
            ),
          if (_isProcessing)
            CircularProgressIndicator(), // Show loading indicator while processing
        ],
      ),
    );
  }
}

class ThirdPage extends StatelessWidget {
  final File image;

  ThirdPage({required this.image});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Heritage Decode',
          style: TextStyle(color: Colors.white),
        ),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(height: 40.0),
          Center(
            child: Image.file(
              image,
              height: 300.0,
              width: 300.0,
            ),
          ),
          SizedBox(height: 20.0),
          Text(
            'Translated Text From The Image',
            style: TextStyle(fontSize: 24.0),
          ),
          SizedBox(height: 40.0),
        ],
      ),
    );
  }
}
