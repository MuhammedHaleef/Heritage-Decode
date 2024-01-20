// main.dart
import 'package:flutter/material.dart';

void main(){
  runApp(MyApp());
}

class MyApp extends StatelessWidget{
  @override
  Widget build(BuildContext context){
    return MaterialApp(
      home: MyHomePage(),

    );
  }
}

// Creating homepage for heritage decode mobile application
class MyHomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Heritage Decode', style: TextStyle(color: Colors.white)),
      ),
      body: Center(
        child: Text('Welcome to Heritage Decode!'),
      ),
    );
  }
}


