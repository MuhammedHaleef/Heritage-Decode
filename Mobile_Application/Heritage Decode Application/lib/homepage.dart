import 'dart:async';
import 'package:flutter/material.dart';
import 'secondpage.dart';

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  void initState() {
    super.initState();
    _navigateToSecondPage();
  }

  _navigateToSecondPage() async {
    Timer(Duration(seconds: 5), () {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => SecondPage()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        height: MediaQuery.of(context).size.height,
        child: Stack(
          fit: StackFit.expand,
          children: [
            Opacity(
              opacity: 1,
              child: Image.asset(
                'assets/3.png', 
                fit: BoxFit.cover,
              ),
            ),
            Positioned(
              bottom: 20.0,
              left: 0.0,
              right: 0.0,
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
                    ),
                    SizedBox(height: 20.0),
                    Text(
                      "Loading...",
                      style: TextStyle(fontSize: 16.0),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
