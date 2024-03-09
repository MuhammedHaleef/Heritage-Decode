import 'package:flutter/material.dart';
// ignore: depend_on_referenced_packages
import 'package:login_page_day_23/dcam.dart'; // Importing the correct file

class TypePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
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
            color: Colors.white,
          ),
        ),
      ),
      body: Container(
        height: MediaQuery.of(context).size.height,
        width: double.infinity,
        child: Stack(
          children: [
            Padding(
              padding: const EdgeInsets.only(bottom: 45),
              //  child: Stepper(
              //        controlsBuilder:...
              //  ),
            ),
            Positioned(
              top: 5,
              left: 0,
              right: 0,
              child: Text(
                "Choose your type of\ncolor blindness",
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 30,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            Positioned(
              left: 50,
              top: 80,
              child: CircleAvatar(
                radius: 60,
                backgroundImage: AssetImage('assets/image1.png'),
              ),
            ),
            Positioned(
              left: 30,
              top: 200,
              child: MaterialButton(
                height: 45,
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => MydPage()), // Corrected navigation
                  );
                },
                color: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(50),
                ),
                child: Text(
                  "Deuteranopia", // Corrected spelling
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 18,
                  ),
                ),
              ),
            ),
            // Add similar Positioned widgets for other options
          ],
        ),
      ),
    );
  }
}
