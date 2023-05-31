import numpy as np
import matplotlib.pyplot as plt
import cv2


def inference(model, test):
    # Load the test image
    test_image = cv2.imread(test)  # Replace with the path to your test image
    # Preprocess the image
    resized_image = cv2.resize(test_image, (128, 128))
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

    predictions = model.predict(input_image)
    predicted_class = None
    
    # Extract the predicted probabilities and bounding box coordinates
    predicted_prob = predictions[0, :, :, 0]
    predicted_boxes = predictions[0, :, :, 1:5]
    predicted_classes = predictions[0, :, :, 5:]

    # Define a threshold for object detection
    threshold = 0.2

    # Create a mask for detected objects above the threshold
    object_mask = predicted_prob > threshold

    # Find the indices of the detected objects
    object_indices = np.where(object_mask)
    # Draw bounding boxes around the detected objects
    for i in range(len(object_indices[0])):
        x1, y1, x2, y2 = predicted_boxes[object_indices[0][i], object_indices[1][i]]
        x1 = int(x1 * 128)  # Convert the coordinates back to the original image size
        y1 = int(y1 * 128)
        x2 = int(x2 * 128)
        y2 = int(y2 * 128)
        predicted_class = np.argmax(predicted_classes[object_indices[0][i], object_indices[1][i]])
        cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        class_label = f"Class: {predicted_class}"
        cv2.putText(test_image, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Display the image with bounding boxes
    plt.axis('off')
    return predicted_class, plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))