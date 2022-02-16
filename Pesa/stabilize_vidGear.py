# import required libraries
from vidgear.gears.stabilizer import Stabilizer
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = cv2.VideoCapture(r'D:\Pesa\Video\test\U.mp4')

size = (int(stream.get(3)),int(stream.get(4)))
result = cv2.VideoWriter("Stabili_vidgear.mp4", 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         30, size)




# initiate stabilizer object with default parameters
stab = Stabilizer()

# loop over
while True:

    # read frames from stream
    (grabbed, frame) = stream.read()
    
    

    # check for frame if not grabbed
    if not grabbed:
        break

    # send current frame to stabilizer for processing
    stabilized_frame = stab.stabilize(frame)
    

    # wait for stabilizer which still be initializing
    if stabilized_frame is None:
        continue

    # {do something with the stabilized frame here}

    # Show output window
    result.write(stabilized_frame)
    cv2.imshow("Stabilized Frame", stabilized_frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

result.release()
# close output window
cv2.destroyAllWindows()

# clear stabilizer resources
stab.clean()

# safely close video stream
stream.release()