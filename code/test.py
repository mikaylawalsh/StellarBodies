import cv2
camera=cv2.VideoCapture(1)

if (camera.isOpened()):
    print("The camera is open")
else:
    print("could not open")

while True: 
    success, frame = camera.read()

    if not success:
        print("not able to read from frame")
        break
    
    cv2.imshow("camera video: ", frame)

    if cv2.waitKey(1) == ord('c'):
        break
    
camera.release()
cv2.destroyAllWindows()
