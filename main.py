import cv2
import dlib
import numpy as np
import time
import tensorflow as tf

# Load the trained CNN model
model = tf.keras.models.load_model('fatigue_detection_model.h5')

predictorFile = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictorFile)

# Open default camera 0
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mouth Aspect Ratio (MAR) calculation
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

EAR_THRESHOLD = 0.25  # if eyes close lower then 0.25 then they are closing their eyes
MAR_THRESHOLD = 0.50  # Value for detecting mouth openness
Blink_Limit = 20 #20 blinks in a minute means fatigue
EyeSleep_base = 10 #if eyes are shut for 10 secs then user is sleeping

# Blink counter
blink_count = 0
blink_detected = False
eye_closure_start_time = None  # Time when eyes first closed
start_time = time.time()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)

        # get the coordinates for the eyes and mouth
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)])
        
        # Calculate EAR and MAR
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0  # Average EAR for both eyes

        mar = mouth_aspect_ratio(mouth)
        
        # Check fatigue with EAR
        # Blink detection logic
        if ear < EAR_THRESHOLD:
            if not blink_detected:  # Only count blink when it was previously open
                blink_count += 1
                blink_detected = True

            if eye_closure_start_time is None:
                eye_closure_start_time = time.time()
            
            # Check if user is sleeping (eyes closed for more than Eyesleep_base
            elapsed_eye_closure_time = time.time() - eye_closure_start_time
            if elapsed_eye_closure_time > EyeSleep_base:
                cv2.putText(frame, "User is Sleeping!", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            blink_detected = False  # Reset flag when eyes open again
            eye_closure_start_time = None

        # Check for fatigue
        elapsed_time = time.time() - start_time
        if elapsed_time > 60:  # Reset counter every 60 seconds
            blink_count = 0
            start_time = time.time()

        # Display blink count and fatigue warning
        cv2.putText(frame, f"Blinks: {blink_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if blink_count >= Blink_Limit:
            cv2.putText(frame, "Fatigue Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Check for yawns
        if mar > MAR_THRESHOLD:
            cv2.putText(frame, "Yawning Detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #CNN MODEL
        # Prepare face region for CNN prediction
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_region = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_region, (128, 128))
        face_normalized = face_resized / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)

        # Predict using the CNN model
        prediction = model.predict(face_expanded)
        fatigue_prediction = 'Fatigue' if prediction[0][0] < 0.5 else 'Alert'
        
        # Display prediction
        confidence = prediction[0][0] * 100  # Convert to percentage
        fatigue_prediction = f"Fatigue ({confidence:.2f}%)" if confidence < 50 else f"Alert ({confidence:.2f}%)"
        cv2.putText(frame, f"Status: {fatigue_prediction}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


        # Draw the face landmarks for eyes
        for n in range(36, 48):  # Eyes landmarks
         cv2.circle(frame, (landmarks.part(n).x, landmarks.part(n).y), 2, (0, 255, 0), -1)

# Draw the face landmarks for mouth
        for n in range(48, 60):  # Mouth landmarks
         cv2.circle(frame, (landmarks.part(n).x, landmarks.part(n).y), 2, (0, 255, 255), -1)


        cv2.circle(frame, (landmarks.part(n).x, landmarks.part(n).y), 2, (0, 255, 0), -1)
        for n in range(48, 60):  # Mouth landmarks
            cv2.circle(frame, (landmarks.part(n).x, landmarks.part(n).y), 2, (0, 255, 255), -1)
        
        # Draw rectangle around the face
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam - Face and Landmark Detection', frame)

    # Break loop with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()