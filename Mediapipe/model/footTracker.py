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
line_count_R_x = 0
line_count_R_x_2 = 0
line_count_R_y = 0
line_count_R_state_y = None

line_count_L_x = 0
line_count_L_x_2 = 0
line_count_L_y = 0
line_count_L_state_y = None

step_list= []

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

    cv2.line(image,(0,400),(width,400),(0,255,0),thickness=5)

    #แตก landmark
    try:
        landmarks = results.pose_landmarks.landmark

        #ข้อเท้าซ้าย
        lx = int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x*width)
        ly = int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y*heigth)
        #แปะจุด
        cv2.circle(image,(lx,ly),10,(255,255,255),cv2.FILLED)
        cv2.line(image,(0,ly),(width+50,ly),(255,255,0),thickness=5)
        
        

        if ly<400:
            if stage_r != 'up' and line_count_L_state_y != 'up':
                line_count_L_x = lx
                line_count_L_y = ly
                print("x(LEFT) up:",line_count_L_x)
                print("y(LEFT) up:",line_count_L_y)
                line_count_L_state_y = 'up'
            stage_l = 'up'
            
        if ly >400 and line_count_L_state_y == 'up' and ly >= ry:
            line_count_L_x_2 = lx
            print("x down:",line_count_L_x_2)
            line_count_L_state_y = 'down'
            result_step_L = abs(line_count_L_x - line_count_L_x_2)
            step_list.append(result_step_L)
            
        if ly>400 and stage_l =='up':
            stage_l= 'down'
            counter_l+=1
            print("L: ",counter_l)

        #ข้อเท้าขวา
        rx = int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x*width)
        ry = int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y*heigth)

        cv2.circle(image,(rx,ry),10,(0,255,0),cv2.FILLED)
        cv2.line(image,(rx,ry),(rx+50,ry),(255,0,0),thickness=5)
        
        #นับการขึ้นของขวา
        if ry<400:
            if stage_l != 'up' and line_count_R_state_y != 'up':
                line_count_R_x = rx
                line_count_R_y = ry
                print("x(RIGHT) up:",line_count_R_x)
                print("y(RIGHT) up:",line_count_R_y)
                line_count_R_state_y = 'up'
            stage_r = 'up'
            
        if ry >400 and line_count_R_state_y == 'up' and ry >= ly:
                line_count_R_x_2 = rx
                print("x down:",line_count_R_x_2)
                line_count_R_state_y = 'down'
                result_step_R = abs(line_count_R_x - line_count_R_x_2)
                step_list.append(result_step_R)
                
            
        if ry>400 and stage_r =='up':
            stage_r= 'down'
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
        print(step_list)
        break
cap.release()
cv2.destroyAllWindows()