import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:bountie/utils/animations.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../data/bg_data.dart';
import '../utils/text_utils.dart';
import 'teacher.dart'; // Import the TeacherScreen
import 'package:firebase_core/firebase_core.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:camera/camera.dart'; // Import the camera package

class LoginScreen extends StatefulWidget {
  final List<CameraDescription> cameras; // Define a variable to hold the cameras

  const LoginScreen({Key? key, required this.cameras}) : super(key: key);

  @override
  State<LoginScreen> createState() => _LoginScreenState(cameras: cameras); // Pass the cameras to the state
}

class _LoginScreenState extends State<LoginScreen> {
  final List<CameraDescription> cameras; // Define a variable to hold the cameras

  _LoginScreenState({required this.cameras}); // Constructor to receive the cameras

  int selectedIndex = 0;
  bool showOption = false;

  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();

  Future<void> loginUser(BuildContext context) async {
    try {
      UserCredential userCredential = await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: emailController.text,
        password: passwordController.text,
      );
      print('User logged in: ${userCredential.user}');

      // Navigate to TeacherScreen upon successful login and pass the cameras
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => TeacherScreen(cameras: cameras)),
      );
    } on FirebaseAuthException catch (e) {
      String errorMessage;
      if (e.code == 'user-not-found') {
        errorMessage = 'No user found for that email.';
      } else if (e.code == 'wrong-password') {
        errorMessage = 'Wrong password provided for that user.';
      } else {
        errorMessage = 'Error: $e';
      }

      // Show error alert
      showDialog(
        context: context,
        builder: (context) {
          return AlertDialog(
            title: Text('Error'),
            content: Text(errorMessage),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context); // Close the alert dialog
                },
                child: Text('OK'),
              ),
            ],
          );
        },
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      floatingActionButton: Container(
        margin: const EdgeInsets.symmetric(vertical: 10),
        height: 49,
        width: double.infinity,
        child: Row(
          children: [
            Expanded(
              child: showOption
                  ? ShowUpAnimation(
                      delay: 100,
                      child: ListView.builder(
                        shrinkWrap: true,
                        itemCount: bgList.length,
                        scrollDirection: Axis.horizontal,
                        itemBuilder: (context, index) {
                          return GestureDetector(
                            onTap: () {
                              setState(() {
                                selectedIndex = index;
                              });
                            },
                            child: CircleAvatar(
                              radius: 30,
                              backgroundColor: selectedIndex == index ? Colors.white : Colors.transparent,
                              child: Padding(
                                padding: const EdgeInsets.all(1),
                                child: CircleAvatar(
                                  radius: 30,
                                  backgroundImage: AssetImage(bgList[index]),
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                    )
                  : const SizedBox(),
            ),
            const SizedBox(width: 20,),
            showOption
                ? GestureDetector(
                    onTap: () {
                      setState(() {
                        showOption = false;
                      });
                    },
                    child: const Icon(Icons.close, color: Colors.white, size: 30),
                  )
                : GestureDetector(
                    onTap: () {
                      setState(() {
                        showOption = true;
                      });
                    },
                    child: CircleAvatar(
                      backgroundColor: Colors.white,
                      child: Padding(
                        padding: const EdgeInsets.all(1),
                        child: CircleAvatar(
                          radius: 30,
                          backgroundImage: AssetImage(bgList[selectedIndex]),
                        ),
                      ),
                    ),
                  )
          ],
        ),
      ),
      body: Container(
        height: double.infinity,
        width: double.infinity,
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage(bgList[selectedIndex]),
            fit: BoxFit.fill,
          ),
        ),
        alignment: Alignment.center,
        child: Container(
          height: 400,
          width: double.infinity,
          margin: const EdgeInsets.symmetric(horizontal: 30),
          decoration: BoxDecoration(
            border: Border.all(color: Colors.white),
            borderRadius: BorderRadius.circular(15),
            color: Colors.black.withOpacity(0.1),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaY: 5, sigmaX: 5),
              child: Padding(
                padding: const EdgeInsets.all(25),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Spacer(),
                    Center(child: TextUtil(text: "Login", weight: true, size: 30)),
                    const Spacer(),
                    TextUtil(text: "Email"),
                    Container(
                      height: 35,
                      decoration: BoxDecoration(border: Border(bottom: BorderSide(color: Colors.white))),
                      child: TextFormField(
                        controller: emailController,
                        style: const TextStyle(color: Colors.white),
                        decoration: const InputDecoration(
                          suffixIcon: Icon(Icons.mail, color: Colors.white),
                          fillColor: Colors.white,
                          border: InputBorder.none,
                        ),
                      ),
                    ),
                    const Spacer(),
                    TextUtil(text: "Password"),
                    Container(
                      height: 35,
                      decoration: BoxDecoration(border: Border(bottom: BorderSide(color: Colors.white))),
                      child: TextFormField(
                        controller: passwordController,
                        obscureText: true,
                        style: const TextStyle(color: Colors.white),
                        decoration: const InputDecoration(
                          suffixIcon: Icon(Icons.lock, color: Colors.white),
                          fillColor: Colors.white,
                          border: InputBorder.none,
                        ),
                      ),
                    ),
                    const Spacer(),
                    Row(
                      children: [
                        Container(
                          height: 15,
                          width: 15,
                          color: Colors.white,
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: TextUtil(text: "Remember Me , FORGET PASSWORD", size: 12, weight: true),
                        )
                      ],
                    ),
                    const Spacer(),
                    GestureDetector(
                      onTap: () => loginUser(context), // Pass the context to loginUser
                      child: Container(
                        height: 40,
                        width: double.infinity,
                        decoration: BoxDecoration(
                            color: Colors.white, borderRadius: BorderRadius.circular(30)),
                        alignment: Alignment.center,
                        child: TextUtil(text: "Log In", color: Colors.black),
                      ),
                    ),
                    const Spacer(),
                    Center(
                        child: TextUtil(
                      text: "Don't have a account REGISTER",
                      size: 12,
                      weight: true,
                    )),
                    const Spacer(),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
