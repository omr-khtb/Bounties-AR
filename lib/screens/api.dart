import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<String> getData(url) async {
 http.Response response = await http.get(Uri.parse(url));
  return response.body;
}