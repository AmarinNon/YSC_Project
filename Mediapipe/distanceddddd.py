import math
import cv2

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


print(calculateDistance(120,25,120,120))
print(calculateDistance(145,170,145,120))
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    width = int(vid.get(3))
    heigth = int(vid.get(4))

    cv2.line(frame,(120,25),(120,120),(255,0,0),thickness=5)
    cv2.line(frame,(145,170),(145,120),(255,0,0),thickness=5)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)

      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()



