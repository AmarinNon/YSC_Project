import cv2
import mediapipe as mp
import numpy as np
import math
import time
# import os
# import csv
from numpy.lib.function_base import angle
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

l_value = []
r_value = []

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist
  
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 0, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size



# For webcam input:
cap = cv2.VideoCapture(0)

# number = input('number:')

# cap = cv2.VideoCapture(r'D:\projectF\Mediapipe\shoulder\shoulder ('+str(number)+').mp4')
# cap = cv2.VideoCapture(r'D:\projectF\Mediapipe\Video\ไม่ป่วย\2\IMG_0117.mp4')
# name_Video = input('Enter Video name:')

# size = (int(cap.get(3)), int(cap.get(4)))
# result = cv2.VideoWriter("shoulder_"+str(number)+"_line.mp4", 
#                          cv2.VideoWriter_fourcc(*'MP4V'),
#                          30, size)

# result = cv2.VideoWriter("R_shoulder_line-2.mp4", 
#                          cv2.VideoWriter_fourcc(*'MP4V'),
#                          30, size)

# result_raw = cv2.VideoWriter(str(name_Video)+"_raw.avi", 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          24, size)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    # frame = image.copy()
    # result_raw.write(image)
    width = int(cap.get(3))
    height = int(cap.get(4))
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

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
        l_distance = calculateDistance(l_hip[0]*width,l_hip[1]*height,l_shoulder[0]*width,l_shoulder[1]*height)
        r_distance = calculateDistance(r_hip[0]*width,r_hip[1]*height,r_shoulder[0]*width,r_shoulder[1]*height)
        

        
        draw_text(image, str(int(l_distance)), font_scale=2, pos=tuple(np.multiply(l_shoulder,[abs(width-(width*0.02)),abs(height-height*0.05)]).astype(int)), text_color_bg=(255, 255, 255))
        # draw_text(image, str(int(r_distance)), font_scale=2, pos=tuple(np.multiply(r_shoulder,[abs(width-(width*0.02)),abs(height-height*0.05)]).astype(int)), text_color_bg=(255, 255, 255))
        
        check_time = time.time()
        l_value.append((int(l_distance)))
        r_value.append((int(r_distance)))
        
        
        # cv2.putText(image,str(int(l_distance)),
        #                 tuple(np.multiply(l_shoulder,[width,height]).astype(int)),
        #                 cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA
        # )
        
        
        # cv2.putText(image,str(int(r_distance)),
        #                 tuple(np.multiply(r_shoulder,[width,height]).astype(int)),
        #                 cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA
        # )

        
    except:
        pass
    #เอาภาพออกมา
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2,circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2,circle_radius=2)
    )
    # result.write(image)
    image = cv2.resize(image, (int(width/2), int(height/2)))
    
    cv2.imshow('Mediapipe', image)
    # cv2.imshow('frame', frame)
    

    if cv2.waitKey(5) & 0xFF == 27:
      break
    
last_value = []
avg_value = 0


for i in range(len(l_value)):
      if l_value[i] > r_value[i]:
            last_value.append((l_value[i] - r_value[i])/l_value[i])
      elif r_value[i] > l_value[i]:
            last_value.append((r_value[i] - l_value[i])/r_value[i])
      else:
            continue

avg_value = sum(last_value) / len(last_value)

print(avg_value)
cap.release()
# result_raw.release()
# result.release()
cv2.destroyAllWindows()



