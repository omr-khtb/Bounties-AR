import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'package:camera/camera.dart'; // Import the camera package
import 'package:bountie/screens/login_screen.dart';
import 'package:bountie/screens/teacher.dart'; // Import the TeacherScreen

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp])
      .then((_) async {
    // Initialize the cameras
    final cameras = await availableCameras();

    runApp(MyApp(cameras: cameras)); // Pass the cameras to MyApp
  });
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({Key? key, required this.cameras}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      initialRoute: '/', // Set initial route
      routes: {
        '/': (context) => LoginScreen(cameras: cameras), // Pass cameras to LoginScreen
        '/teacher': (context) => TeacherScreen(cameras: cameras), // Pass cameras to TeacherScreen
      },
    );
  }
}

