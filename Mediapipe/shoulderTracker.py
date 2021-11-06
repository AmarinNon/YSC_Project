import cv2
import mediapipe as mp
import numpy as np
import math
from numpy.lib.function_base import angle
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist



# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    width = int(cap.get(3))
    heigth = int(cap.get(4))
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #make detection
    results = pose.process(image)

    # Draw the pose annotation on the image. also make image to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #แตก landmark
    try:
        landmarks = results.pose_landmarks.landmark

        #เลือกส่วนของร่างกาย
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
      
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        #คำนวณระยะ
        l_distance = calculateDistance(l_hip[0]*width,l_hip[1]*heigth,l_shoulder[0]*width,l_shoulder[1]*heigth)
        r_distance = calculateDistance(r_hip[0]*width,r_hip[1]*heigth,r_shoulder[0]*width,r_shoulder[1]*heigth)
        

        

        cv2.putText(image,str(int(l_distance)),
                        tuple(np.multiply(l_shoulder,[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA
        )
        cv2.putText(image,str(int(r_distance)),
                        tuple(np.multiply(r_shoulder,[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA
        )

        
    except:
        pass

    

    
  
    #เอาภาพออกมา
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2,circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2,circle_radius=2)
    
    
    )
    
    cv2.imshow('Mediapipe', image)
    

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()