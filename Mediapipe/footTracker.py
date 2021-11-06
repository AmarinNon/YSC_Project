import cv2
import mediapipe as mp
import numpy as np
from numpy.lib.function_base import angle
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


counter_l = 0
stage_l = None
counter_r = 0
stage_r = None

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

    cv2.line(image,(0,420),(width,420),(0,255,0),thickness=5)

    #แตก landmark
    try:
        landmarks = results.pose_landmarks.landmark

        #ข้อเท้าซ้าย
        lx = int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x*width)
        ly = int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y*heigth)
        #แปะจุด
        cv2.circle(image,(lx,ly),10,(255,0,255),cv2.FILLED)

        if ly<420:
            stage_l = 'down'
        if ly>420 and stage_l =='down':
            stage_l= 'up'
            counter_l+=1
            print("L: ",counter_l)

        #ข้อเท้าขวา
        rx = int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x*width)
        ry = int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y*heigth)

        cv2.circle(image,(rx,ry),10,(255,0,255),cv2.FILLED)

        if ry<420:
            stage_r = 'down'
        if ry>420 and stage_r =='down':
            stage_r= 'up'
            counter_r+=1
            print("R: ",counter_r)


        
        
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