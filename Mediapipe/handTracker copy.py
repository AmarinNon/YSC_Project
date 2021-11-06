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
delay = 3.0
timer = 15.0 + delay
checkStart = False


def cal_angle(a,b,c):
          a = np.array(a)
          b = np.array(b)
          c = np.array(c)

          radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
          angle = np.abs(radians*180.0/np.pi)

          if angle > 180.0:
              angle = 360-angle
          return angle






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
        cv2.putText(image,str(angle),
                tuple(np.multiply(Wrist,[640,480]).astype(int)),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)

        for id,lm in enumerate(hand_landmarks.landmark):
              h,w,c = image.shape
              cx,cy =int(lm.x * w),int(lm.y * h)


              if id==5:
                    cv2.circle(image,(cx,cy),5,(255,0,0),cv2.FILLED)
                    
                    if tstart == "NO" and cy>225 and cy <255:
                          start_time = time.time()
                          print("Hi")
                          tstart = "YES"
                          
                    if tstart != "NO" and time.time() - start_time >= delay and checkStart == False:
                          checkStart = True
                          angle_before_start = angle
                          print(angle_before_start)
                          
                    if tstart != "NO" and time.time() - start_time > delay :
                          x = angle_before_start
                          cv2.putText(image,"Start",(width-120,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 145, 255),2,cv2.LINE_AA)
                          print(abs(x-angle))
                          if abs(x-angle) <= 10:
                                stage = 'default'
                          if abs(x-angle) > 10 and stage =='default':
                                stage = 'change'
                                counter+=1
                                print("new:",counter)




    
                          

                          

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