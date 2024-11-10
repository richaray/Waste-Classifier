import cv2
import argparse
from ultralytics import YOLO

# Class ID for 'person' in the COCO dataset
PERSON_CLASS_ID = 0
strings_list = []
i=0

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def wastee():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load YOLOv8 model
    model = YOLO("yolov8l.pt")

    # Disable verbose YOLOv8 logging
    model.overrides['verbose'] = False


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run inference with YOLOv8 model
        result = model(frame)[0]

        # Extract bounding boxes, class IDs, and confidences
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs (converted to int)

        # Filter out the 'person' class (class ID 0)
        filtered_boxes = []
        for i, class_id in enumerate(class_ids):
            if class_id != PERSON_CLASS_ID:  # Exclude 'person' class
                filtered_boxes.append((boxes[i], confidences[i], class_id))

        # Draw filtered bounding boxes
        frame_with_filtered_boxes = frame.copy()
        for box, confidence, class_id in filtered_boxes:
            x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
            label = f"{model.names[class_id]} {confidence:.2f}"  # Class label with confidence

            # Draw the bounding box
            cv2.rectangle(frame_with_filtered_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the class label and confidence
            cv2.putText(frame_with_filtered_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print only non-person detections in the terminal
            #print(f"Detected: {label}")
            strings_list.append(label)
            i=i+1
            # Simulating object detection process
            # label = "bottle"  # Example label from detection logic

            # print(f"Detected: {label}")


        # Show the filtered frame
        cv2.imshow("yolov8", frame_with_filtered_boxes)

        # Break the loop if 'Esc' is pressed (ASCII code 27)
        if cv2.waitKey(30) == 27:
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    return strings_list[i]
# Run the program

