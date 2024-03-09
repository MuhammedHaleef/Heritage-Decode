import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class MydPage extends StatefulWidget {
  @override
  _MydPageState createState() => _MydPageState();
}

class _MydPageState extends State<MydPage> {
  late File? imageFile; // Marking as late and nullable

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color.fromARGB(255, 43, 43, 44),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Color.fromARGB(255, 43, 43, 44),
        leading: IconButton(
          onPressed: () {
            Navigator.pop(context);
          },
          icon: Icon(
            Icons.arrow_back_ios,
            size: 20,
            color: Color.fromARGB(255, 255, 255, 255),
          ),
        ),
        title: Text("Upload Image"),
      ),
      body: Stack(
        children: <Widget>[
          imageFile == null
              ? Container(
                  alignment: Alignment.center,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                            foregroundColor: Colors.blueAccent,
                            backgroundColor: Colors.white30,
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(20))),
                        onPressed: () {
                          _getFromGallery();
                        },
                        child: Text("CHOOSE FROM GALLERY"),
                      ),
                      Container(
                        height: 60.0,
                      ),
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          foregroundColor: Colors.blueAccent,
                          backgroundColor: Colors.white30,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(20),
                          ),
                        ),
                        onPressed: () {
                          _getFromCamera();
                        },
                        child: Text("CHOOSE FROM CAMERA"),
                      )
                    ],
                  ),
                )
              : Container(
                  child: Image.file(
                    imageFile!,
                    fit: BoxFit.cover,
                  ),
                ),
          Positioned(
            bottom: 20,
            left: 0,
            right: 0,
            child: imageFile == null
                ? SizedBox.shrink()
                : ElevatedButton(
                    onPressed: () {
                      // Implement your upload logic here
                    },
                    child: Text("UPLOAD"),
                  ),
          ),
        ],
      ),
    );
  }

  /// Get from gallery
  _getFromGallery() async {
    final pickedFile = await ImagePicker().getImage(
      source: ImageSource.gallery,
      maxWidth: 1400,
      maxHeight: 1400,
    );
    if (pickedFile != null) {
      setState(() {
        imageFile = File(pickedFile.path);
      });
    }
  }

  /// Get from Camera
  _getFromCamera() async {
    final pickedFile = await ImagePicker().getImage(
      source: ImageSource.camera,
      maxWidth: 1400,
      maxHeight: 1400,
    );
    if (pickedFile != null) {
      setState(() {
        imageFile = File(pickedFile.path);
      });
    }
  }
}
