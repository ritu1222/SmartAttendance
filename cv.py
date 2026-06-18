import cv2
from cv2 import VideoCapture
cap = cv2.VideoCapture(0)
while True:

  ret, frame = cap.read()
  cv2.imshow('WEBCAM', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  key = cv2.waitKey(1)

  if key == ord('s'):
    cv2.imwrite(
      'webcam_capture.jpg', frame
    )

    print('Image saved as webcam_capture.jpg')
    
cap.release()
cv2.destroyAllWindows()