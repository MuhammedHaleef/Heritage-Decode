import 'package:flutter/material.dart';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Heritage Decode', style: TextStyle(color: const Color.fromARGB(255, 106, 90, 90))),
        leading: IconButton(
          icon: Icon(Icons.menu),
          onPressed: () {},
        ),
        backgroundColor: const Color.fromARGB(255, 62, 119, 167),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Center(
            child: Image.asset(
              'assets/Logo.png',
              height: 300.0,
              width: 300.0,
            ),
          ),
          SizedBox(height: 20.0),
          OutlinedButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => SecondPage()),
              );
            },
            child: Text(
              'Get Started',
              style: TextStyle(fontSize: 20.0),
            ),
            style: OutlinedButton.styleFrom(
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

  Future<void> _getImageFromCamera() async {
    final picker = ImagePicker();
    final pickedFile = await picker.getImage(source: ImageSource.camera);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
      });
    }
  }

  Future<void> _getImageFromFileChooser() async {
    final picker = ImagePicker();
    final pickedFile = await picker.getImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
      });
    }
  }

  Future<String?> _getMajorColor(File image) async {
    var request = http.MultipartRequest('POST', Uri.parse('http://127.0.0.1:5000'));
    request.files.add(await http.MultipartFile.fromPath('file', image.path));

    var response = await request.send();
    if (response.statusCode == 200) {
      var jsonResponse = await response.stream.bytesToString();
      return jsonDecode(jsonResponse)['major_color'];
    } else {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('This is the Second Page', style: TextStyle(color: Colors.white)),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        backgroundColor: const Color.fromARGB(255, 62, 119, 167),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (_selectedImage != null)
            Center(
              child: Image.file(
                _selectedImage!,
                height: 350.0,
                width: 350.0,
              ),
            ),
          SizedBox(height: 20.0),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              IconButton(
                icon: Icon(Icons.camera),
                onPressed: () async {
                  await _getImageFromCamera();
                },
              ),
              IconButton(
                icon: Icon(Icons.file_upload),
                onPressed: () async {
                  await _getImageFromFileChooser();
                },
              ),
            ],
          ),
          SizedBox(height: 20.0),
          if (_selectedImage != null)
            ElevatedButton(
              onPressed: () async {
                String? majorColor = await _getMajorColor(_selectedImage!);
                if (majorColor != null) {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => ThirdPage(image: _selectedImage!, majorColor: majorColor),
                    ),
                  );
                } else {
                  // Handle error
                }
              },
              child: Text('Convert'),
            ),
        ],
      ),
    );
  }
}

class ThirdPage extends StatelessWidget {
  final File image;
  final String majorColor;

  ThirdPage({required this.image, required this.majorColor});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('This is the Third Page', style: TextStyle(color: Colors.white)),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        backgroundColor: const Color.fromARGB(255, 62, 119, 167),
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
            'Translated Image',
            style: TextStyle(fontSize: 24.0),
          ),
          SizedBox(height: 20.0),
          Text(
            'Major Color: $majorColor',
            style: TextStyle(fontSize: 18.0),
          ),
          SizedBox(height: 40.0),
        ],
      ),
    );
  }
}
