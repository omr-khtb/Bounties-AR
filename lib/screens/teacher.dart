import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'teacher_camera.dart'; // Import the CameraPage
import 'login_screen.dart'; // Assuming you have a LoginScreen widget
import 'package:firebase_auth/firebase_auth.dart'; // Assuming you're using Firebase Auth
import 'dart:convert';
import 'package:bountie/screens/api.dart';

class TeacherScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const TeacherScreen({Key? key, required this.cameras}) : super(key: key);

  @override
  State<TeacherScreen> createState() => TeacherScreenState();
}

class TeacherScreenState extends State<TeacherScreen> {
  String? url;
  var data;
  String queryText = 'Query';

  void _logout(BuildContext context) async {
    await FirebaseAuth.instance.signOut();
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => LoginScreen(cameras: widget.cameras)),
    );
  }

  void _goToCamera(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => CameraPage(camera: widget.cameras.first),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('PYTHON AND FLUTTER'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          Padding(
            padding: const EdgeInsets.all(10.0),
            child: TextField(
              onChanged: (value) {
                setState(() {
                  url = 'http://10.0.2.2:5000/api?ID=2&Role=teacher';
                });
              },
              decoration: InputDecoration(
                hintText: 'Search Anything Here',
                suffixIcon: GestureDetector(
                  onTap: () async {
                    data = await getData(url);
                    var decodedData = jsonDecode(data);
                    setState(() {
                      queryText = decodedData['Answer'];
                    });
                  },
                  child: Icon(Icons.search),
                ),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(10.0),
            child: Text(
              queryText,
              style: TextStyle(fontSize: 30.0, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
          ),
          ElevatedButton(
            onPressed: () => _goToCamera(context),
            child: Text('Open Camera'),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => _logout(context),
        child: Icon(Icons.logout),
      ),
    );
  }
}
