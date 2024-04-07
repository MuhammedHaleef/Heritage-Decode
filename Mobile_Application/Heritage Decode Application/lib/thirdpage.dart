import 'dart:io';
import 'package:flutter/material.dart';

class ThirdPage extends StatelessWidget {
  final File selectedImage;
  final String predictedClass;

  const ThirdPage({Key? key, required this.selectedImage, required this.predictedClass})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/background.jpg"), // Replace with your image path
            fit: BoxFit.cover,
            colorFilter: ColorFilter.mode(
              Colors.black.withOpacity(0.7), // Adjust opacity here
              BlendMode.dstATop,
            ),
          ),
        ),
        child: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start, // Align widgets to the top
              crossAxisAlignment: CrossAxisAlignment.center,
              children: <Widget>[
                const SizedBox(height: 20),
                // Display selected image
                Container(
                  width: 300, // Decreased width
                  height: 250, // Decreased height
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: Colors.blue,
                      width: 2,
                    ),
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Image.file(
                    selectedImage,
                    width: 200,
                    height: 200,
                    fit: BoxFit.cover,
                  ),
                ),
                const SizedBox(height: 20),
                // Display translated text label with outlined text
                Text(
                  'Translated Text:',
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Colors.white, // Text color set to white
                    shadows: <Shadow>[
                      Shadow(
                        offset: Offset(-1.5, -1.5),
                        color: Colors.black,
                      ),
                      Shadow(
                        offset: Offset(1.5, -1.5),
                        color: Colors.black,
                      ),
                      Shadow(
                        offset: Offset(1.5, 1.5),
                        color: Colors.black,
                      ),
                      Shadow(
                        offset: Offset(-1.5, 1.5),
                        color: Colors.black,
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 10),
                // Display translated text output
                Card(
                  elevation: 4,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                    side: BorderSide(
                      color: Colors.blue,
                      width: 2,
                    ),
                  ),
                  child: Container(
                    width: 300,
                    padding: EdgeInsets.all(10),
                    child: Text(
                      '$predictedClass',
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
                const SizedBox(height: 40),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
