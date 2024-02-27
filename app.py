from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import json
import os
import random
import string
import mediapipe as mp
import base64

# object detection and save
def detect_objects(image):
    config_file = "yolov3.cfg"
    weights_file = "yolov3.weights"
    classes_file = "yolov3.txt"

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights_file, config_file)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    detected_objects = set()  # Set to store detected object names

    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                detected_objects.add(classes[class_id])  # Add detected object name to the set

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    cv2.imshow("object detection", image)
    cv2.waitKey()
    cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()

    return detected_objects


def save(detected_objects, image):
    # Generate random ID
    random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

    # Save detected objects and image ID to JSON file
    json_file_path = "bounties.json"
    if os.path.exists(json_file_path) and os.path.getsize(json_file_path) > 0:
        # If the file exists and is not empty, load its contents
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
            # Update the existing data with the new data
            json_data[random_id] = {'detected_objects': list(detected_objects), 'image_id': random_id}
    else:
        # If the file doesn't exist or is empty, create a new JSON data structure
        json_data = {random_id: {'detected_objects': list(detected_objects), 'image_id': random_id}}

    # Write the updated data back to the file
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    # Save original image with random ID
    cv2.imwrite(f"{random_id}.jpg", image)


# similarityyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy

def check_similarity(image1, image2):
    # Load images
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Initialize feature detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Initialize feature matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Set a threshold for the number of matches
    threshold = 10

    # Check if the number of matches is above the threshold
    if len(matches) >= threshold:
        return len(matches)
    else:
        return False


# thumbs up detectionnnnnnnnnnnnn


def detect_thumbs_up(image=None):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    thumbs_up_detected = False
    thumbs_up_frame = None  # Initialize variable to store the frame with thumbs-up gesture

    if image is not None:
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                try:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[
                        mp_hands.HandLandmark.THUMB_MCP].y:
                        thumbs_up_detected = True
                        thumbs_up_frame = image.copy()  # Save the frame with thumbs-up gesture
                        return thumbs_up_frame
                except Exception as e:
                    pass
    else:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    try:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[
                            mp_hands.HandLandmark.THUMB_MCP].y:
                            thumbs_up_detected = True
                            thumbs_up_frame = frame.copy()  # Save the frame with thumbs-up gesture
                            break
                    except Exception as e:
                        pass

            cv2.imshow('Thumbs Up Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or thumbs_up_detected:
                break

        cap.release()
        cv2.destroyAllWindows()

    return thumbs_up_frame if thumbs_up_detected else None


def load_data_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)


def compare_objects(detected_objects, json_data):
    matched_entries = []
    for entry_id, entry_data in json_data.items():
        for obj in entry_data['detected_objects']:
            for obj2 in detected_objects:
                if obj == obj2:
                    matched_entries.append(entry_id)
    return matched_entries


def check_similarity_for_matched_entries(matched_entries, image_from_detection):
    similarity_list = []
    for entry_id in matched_entries:
        image_path = f"{entry_id}.jpg"
        if os.path.exists(image_path):
            similarit_rate = check_similarity(image_from_detection, image_path)
            if similarit_rate:
                this_similarity = []
                this_similarity.append(image_path)
                this_similarity.append(similarit_rate)
                similarity_list.append(this_similarity.copy())

    return similarity_list


def SendMeImage(role, image=None):
    if image is not None:
        image = detect_thumbs_up(image)
        if image is not None:
            image_copy = image.copy()
            detected_objects = detect_objects(image_copy)
            if role == "teacher":
                print("teacher")
                save(detected_objects, image)
                print("Bounty Saved Teacher")
                return "Bounty Saved Teacher"
            else:
                json_file_path = "bounties.json"
                json_data = load_data_from_json(json_file_path)
                cv2.imwrite("thisTry.jpg", image)
                matched_entries = compare_objects(detected_objects, json_data)
                print("matched objects: ", matched_entries)
                checks_similarity = check_similarity_for_matched_entries(matched_entries, "thisTry.jpg")
                print("matched similarity: ", checks_similarity)
                if matched_entries and checks_similarity:
                    print("Done You Found Bounty")
                    return "Done You Found Bounty"
                else:
                    print("No matched bounties found in the JSON file.")
                    return "No matched bounties found in the JSON file."

        else:
            print("No thumbs up detected.")
            return "No thumbs up detected."
    else:
        print("No image detected.")
        return "No image detected."


def letMeGetImage(role):
        image = detect_thumbs_up()
        if image is not None:
            image_copy = image.copy()
            detected_objects = detect_objects(image_copy)
            if role == "teacher":
                print("teacher")
                save(detected_objects, image)
                print("Done")
            else:
                json_file_path = "bounties.json"
                json_data = load_data_from_json(json_file_path)
                cv2.imwrite("thisTry.jpg", image)
                matched_entries = compare_objects(detected_objects, json_data)
                checks_similarity = check_similarity_for_matched_entries(matched_entries, "thisTry.jpg")
                print("matched objects: ", matched_entries)
                print("matched similarity: ", checks_similarity)
                if matched_entries and checks_similarity:
                    # to delete the bountie
                    print("Done predicted")
                else:
                    print("No matched entries found in the JSON file.")
        else:
            print("No thumbs up detected.")


#input_choice = input("Teacher Or Student ?")
#image_path = 'object-detection.jpg'
#image = cv2.imread(image_path)
# letMeGetImage(input_choice)
#SendMeImage(input_choice, image)


app = Flask(__name__)


@app.route('/api', methods=['GET'])
def hello_world():
    d = {}
    d["ID"] = str(request.args['ID']) + ".jpg"
    d["Role"] = str(request.args['Role'])
    if d['Role'] == 'teacher':
        image_path = d["ID"]
        image = cv2.imread(image_path)
        d["Answer"] = SendMeImage("teacher", image)
    return jsonify(d)


@app.route('/api', methods=['POST'])
def process_image():
    # Receive the image data from the request body
    data = request.get_json()
    image_data = data['image']

    # Decode the base64 image data
    image_bytes = base64.b64decode(image_data)

    # Convert the image bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the image array to an OpenCV image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image (detect objects, etc.)
    # For example:
    # detected_objects = detect_objects(image)
    # save(detected_objects, image)

    # Here you can call your existing functions to process the image
    # and return the result as needed

    # Return a response (you can customize this based on your application's requirements)
    response = {'message': 'Image received and processed successfully'}
    return jsonify(response)



if __name__ == '__main__':
    app.run()
