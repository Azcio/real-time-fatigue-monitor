# real-time-fatigue-monitor
Developed a real-time fatigue detection system that uses computer vision to monitor facial features via webcam and predict user fatigue levels. The system employs OpenCV for live face and eye tracking, processes image-based indicators (like blink rate and eye aspect ratio), and leverages a trained machine learning model to classify fatigue levels. The pipeline includes data preprocessing, model training, and real-time prediction, achieving high accuracy and responsiveness. Designed modularly with clear separation between data preparation, model training, and live application logic.

Achievements:
Achieved 1st class for this project in a Computer Vision/Machine Learning course
Built a fully functional pipeline from data acquisition to model inference
Demonstrated strong skills in Python, OpenCV, and ML model integration in real-world scenarios

How to run:
- Python 3.x installed
- Webcam connected
- Required Python packages installed

- first run LoadData.py file
- Then run fatigueModel.py file (will take about 2-5 mins)
- Finally run main.py (Make sure camera is connected and not in use, to stop running press "Q")
