import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String message = '';

  void fetchData() async {
    final response = await http.get(Uri.parse('http://localhost:5000/api/data'));
    if (response.statusCode == 200) {
      setState(() {
        message = jsonDecode(response.body)['message'];
      });
    } else {
      setState(() {
        message = 'Failed to fetch data';
      });
    }
  }

  void postData() async {
    final response = await http.post(
      Uri.parse('http://localhost:5000/api/post_data'),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, String>{'data': 'Some data from Flutter'}),
    );
    if (response.statusCode == 200) {
      setState(() {
        message = jsonDecode(response.body)['message'];
      });
    } else {
      setState(() {
        message = 'Failed to post data';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Flutter Demo'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              'Message from backend:',
            ),
            Text(
              '$message',
              style: TextStyle(fontSize: 20),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: fetchData,
              child: Text('Fetch Data'),
            ),
            ElevatedButton(
              onPressed: postData,
              child: Text('Post Data'),
            ),
          ],
        ),
      ),
    );
  }
}
