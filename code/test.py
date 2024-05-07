import cv2
camera=cv2.VideoCapture(0)

if (camera.isOpened()):
    print("The camera is open")
else:
    print("could not open")

for i in range(100): 
    success, frame = camera.read()

    if not success:
        print("not able to read from frame")
        break
    
    cv2.imshow("camera video: ", frame)

    if cv2.waitKey(1) == ord('c'):
        break

success, frame = camera.read()

if not success:
    print("not able to read from frame")

cv2.imwrite("sample.png", frame)
camera.release()
cv2.destroyAllWindows()
