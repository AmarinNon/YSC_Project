import cv2
import mediapipe as mp
import numpy as np

import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

counter = 0
stage = None
tstart = "NO"
end =0
timer = 10





cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5,max_num_hands=1) as hands:
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


    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.line(image,(0,240),(width,240),(0,0,255),thickness=5)
    


    cv2.line(image,(0,255),(width,255),(0,255,0),thickness=5)
    cv2.line(image,(0,225),(width,225),(255,0,0),thickness=5)

    cv2.putText(image,str(counter),(30,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
    
    


    # cv2.circle(image,(0,225),15,(255,255,255),cv2.FILLED)
    
    # cv2.line(image,(100,100),(600,400),(0,255,0),thickness=5)
    # cv2.line(image,(100,400),(600,100),(0,255,0),thickness=5)
    # cv2.line(image,(0,0),(width,heigth),(0,255,0),thickness=5)
    # cv2.line(image,(0,heigth),(width,0),(0,255,0),thickness=5)





    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2,circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2,circle_radius=2)
            )
        for id,lm in enumerate(hand_landmarks.landmark):
              h,w,c = image.shape
              cx,cy =int(lm.x * w),int(lm.y * h)
              # print("W:",w)

              if id==5:
                    cv2.circle(image,(cx,cy),5,(255,0,0),cv2.FILLED)
                    if cy<225:
                      stage = 'down'
                    if cy>225 and stage =='down':
                      stage= 'up'
                      counter+=1
                      print(counter)
                    if tstart == "NO" and cy>225 and cy <255:
                          start_time = time.time()
                          print("Hi")
                          tstart = "YES"


    
                          

                          

              # if id ==9:
              #       cv2.circle(image,(cx,cy),5,(0,255,0),cv2.FILLED)
              # if id ==13:
              #       cv2.circle(image,(cx,cy),5,(0,0,255),cv2.FILLED)
              # if id ==17:
              #       cv2.circle(image,(cx,cy),5,(255,255,0),cv2.FILLED)
                    

    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if tstart == "YES" and time.time()-start_time >= timer:
          print("end")
          end = 1
    cv2.imshow('MediaPipe Hands', (image))
    if cv2.waitKey(5) & 0xFF == 27 or end == 1:
      break
cap.release()
print("Hz = ",counter/timer)