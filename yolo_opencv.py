import cv2
import numpy as np
import json
import os
import random
import string

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
    cv2.imwrite(f"{random_id}_original.jpg", image)





image = cv2.imread("1.jpg")
image_copy = image.copy()
detected_objects = detect_objects(image_copy)
print("Detected Objects:", detected_objects)

save(detected_objects, image)