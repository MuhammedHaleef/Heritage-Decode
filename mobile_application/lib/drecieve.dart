import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class ImageScreen extends StatefulWidget {
  @override
  _ImageScreenState createState() => _ImageScreenState();
}

class _ImageScreenState extends State<ImageScreen> {
  late Uint8List imageData;

  Future<void> getImage() async {
    var response =
        await http.get(Uri.parse('http://192.168.1.104:8000/api/receive'));
    if (response.statusCode == 200) {
      setState(() {
        imageData = response.bodyBytes;
      });
    } else {
      throw Exception('Failed to load image');
    }
  }

  @override
  void initState() {
    super.initState();
    getImage();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Received Image'),
      ),
      body: imageData.isEmpty
          ? Center(child: CircularProgressIndicator())
          : Image.memory(
              imageData,
              fit: BoxFit.cover,
            ),
    );
  }
}

void main() {
  runApp(MaterialApp(
    home: ImageScreen(),
  ));
}
