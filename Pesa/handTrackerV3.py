import cv2
import mediapipe as mp
import numpy as np
import csv 
import time
import os
from moviepy.editor import VideoFileClip

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

counter = 0
stage = None
tstart = "NO"
timer = 15.0
checkStart = False
delay = 2
check_time = None



def cal_angle(a,b,c):
          a = np.array(a)
          b = np.array(b)
          c = np.array(c)

          radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
          angle = np.abs(radians*180.0/np.pi)

          if angle > 180.0:
              angle = 360-angle
          return angle



cap = cv2.VideoCapture(r'D:\projectF\part11.png')







down_line = int((int(cap.get(4)) / 2) + 20)
upper_line = int((int(cap.get(4)) / 2) - 20)





with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5,max_num_hands=2) as hands:
  while cap.isOpened():
    success, image = cap.read()
#     result_raw.write(image)
    # frame = image.copy()
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


    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.line(image,(0,240),(width,240),(0,0,255),thickness=5)
    


#     cv2.line(image,(0,down_line),(width,down_line),(0,255,0),thickness=5)
#     cv2.line(image,(0,upper_line),(width,upper_line),(255,0,0),thickness=5)
    

    # cv2.putText(image,str(counter),(30,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)


    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        Thumb_cmc = [hand_landmarks.landmark[1].x,hand_landmarks.landmark[1].y]
        Wrist = [hand_landmarks.landmark[0].x,hand_landmarks.landmark[0].y]
        Index_finger_mcp = [hand_landmarks.landmark[5].x,hand_landmarks.landmark[5].y]

        angle = cal_angle(Thumb_cmc,Wrist,Index_finger_mcp)

        
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2,circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2,circle_radius=2)
            )
        # cv2.putText(image,str(angle),
        #         tuple(np.multiply(Wrist,[width,height]).astype(int)),
        #         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)

        for id,lm in enumerate(hand_landmarks.landmark):
              h,w,c = image.shape
              cx,cy =int(lm.x * w),int(lm.y * h)
              if id==5:
                    cv2.circle(image,(cx,cy),5,(255,0,0),cv2.FILLED)
                    
                    




    
                          
                    
    image = cv2.resize(image, (int(width/2), int(height/2)))
    cv2.imwrite("Mediapipe_hand.png",image)
    cv2.imshow('MediaPipe Hands',image)
    # cv2.imshow('Frame', frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break
# result_raw.release()
# result.release()
cap.release()
cv2.destroyAllWindows()
# print("Hz = ",counter/(clip.duration))

# dirname = os.path.dirname(os.path.abspath(__file__))
# csvfilename = os.path.join(dirname, 'MyCSVFile.csv')
# file_exists = os.path.isfile(csvfilename)
# with open(csvfilename, 'a',newline='') as csvfile: 
#     fields = ['Hz'] 
#     csvwriter = csv.DictWriter(csvfile,fieldnames=fields) 
#     if not file_exists:
#       csvwriter.writeheader()
#     csvwriter.writerow({'Hz':counter/(clip.duration)})
          