import cv2
import supervision as sv
from ultralytics import YOLOv10
import pyttsx3
import pytesseract
from PIL import Image
import threading
import firebase_admin
from firebase_admin import credentials, db

# Initialize the YOLO model
model = YOLOv10('best.pt')
pytesseract.pytesseract.tesseract_cmd = r'D:\PythonTesseractTextReading\tesseract.exe'

# Firebase configuration
cred = credentials.Certificate("firebaseCredentials.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://hackathon-b9731-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set speech rate (default is usually around 200 words per minute, adjust as needed)
rate = engine.getProperty('rate')  # Get the current speech rate
engine.setProperty('rate', rate - 50)  # Decrease the speech rate by 50

# Flag to log results structure
logged_results_structure = False

# Variable to store detected classes
detected_classes = []

# Event to stop the TTS engine
stop_tts_event = threading.Event()
tts_lock = threading.Lock()

# Variable to store the remaining text to be spoken
remaining_text = None


def speak(text):
    global stop_tts_event
    with tts_lock:
        if not stop_tts_event.is_set():
            engine.say(text)
            engine.runAndWait()

# Function to fetch data from Firebase


def fetch_data_from_firebase():
    ref = db.reference('/distance')
    data = ref.get()
    print("Data from Firebase:", data)
    return data


# Loop to continuously get frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform object detection
    try:
        results = model(frame)

        # Log the structure of the results object
        if not logged_results_structure:
            print("Results structure:", type(results), results)
            logged_results_structure = True

        # Handle different formats of results
        if isinstance(results, list) and len(results) > 0:
            predictions = results[0]
        elif isinstance(results, dict):
            predictions = results['pred'][0] if 'pred' in results else None
        else:
            predictions = None

        if predictions is not None:
            detections = sv.Detections.from_ultralytics(predictions)
            detected_classes = [detection[-1]['class_name']
                                for detection in detections if detection[-1] and 'class_name' in detection[-1]]

            # Annotate the frame
            annotated_frame = box_annotator.annotate(
                scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections)

            # Display the resulting frame
            cv2.imshow('Webcam', annotated_frame)
            
        else:
            print("No valid predictions found in results.")
            detected_classes = []

    except Exception as e:
        print("Error during processing:", e)
        break

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # Announce detected classes
    if key == ord('w'):
        stop_tts_event.clear()
        if detected_classes:
            for class_name in detected_classes:
                print(f"Detected class: {class_name}")
                if class_name=='unknown person':
                    tts_thread = threading.Thread(
                    target=speak, args=(class_name + " is nearby you stay alert ",))
                    tts_thread.start()
                    data = fetch_data_from_firebase()
                else:
                    tts_thread = threading.Thread(
                    target=speak, args=(class_name + " is nearby you",))
                    tts_thread.start()
                    data = fetch_data_from_firebase()    
        
        if (type(data) == float or type(data) == int):
            tts_thread = threading.Thread(
                target=speak, args=("An object is at "+str(data)+"meter please move left a little bit for safe walk",))
            tts_thread.start()
            
        else:
            print("No objects detected.")
            tts_thread = threading.Thread(
                target=speak, args=("No objects detected.",))
            tts_thread.start()

    # Capture image and extract text
    elif key == ord('r'):
        stop_tts_event.clear()
        captured_image_path = "captured_frame.jpg"
        cv2.imwrite(captured_image_path, frame)
        print("Image captured and saved.")

        # Extract text from the captured image using Tesseract OCR
        try:
            image = Image.open(captured_image_path)
            extracted_text = pytesseract.image_to_string(image)
            print("Extracted Text:\n", extracted_text)

            # Provide a summary of the extracted text
            if extracted_text:
                summary = "Text detected in the image. Here is a brief summary. " + \
                    extracted_text[:300]
                remaining_text = extracted_text[600:]
                tts_thread = threading.Thread(target=speak, args=(summary,))
                tts_thread.start()
            else:
                print("No text detected in the image.")
                tts_thread = threading.Thread(
                    target=speak, args=("No text detected in the image.",))
                tts_thread.start()

        except Exception as e:
            print("Error during text extraction:", e)
            tts_thread = threading.Thread(
                target=speak, args=("Error during text extraction.",))
            tts_thread.start()

    # Continue reading more text if 'm' is pressed
    elif key == ord('m') and remaining_text:
        stop_tts_event.clear()
        tts_thread = threading.Thread(target=speak, args=(remaining_text,))
        tts_thread.start()
        remaining_text = None

    #Fetch data from Firebase
    elif key == ord('f'):
        data = fetch_data_from_firebase()
        print(data)
        if (type(data) == float or type(data) == int):
            tts_thread = threading.Thread(target=speak, args=("object at "+str(data)+"meter",))
            tts_thread.start()

        else:
            tts_thread = threading.Thread(
                target=speak, args=("no object near you its a clear path to walk straight",))
            tts_thread.start()
        

    # Stop TTS engine
    elif key == ord('s'):
        with tts_lock:
            stop_tts_event.set()
            engine.stop()
            print("Stopped reading text.")
            remaining_text = None

    # Break the loop on 'q' key press
    elif key == ord('q'):
        break

# When everything done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
