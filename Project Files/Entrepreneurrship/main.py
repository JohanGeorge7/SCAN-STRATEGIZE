import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "C:/Users/kavis/Desktop/School Stuff/code go brrr/VSCOdE/Entrepreneurrship/data/cascade.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(1)

while True:
  _, frame = camera.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = clf.detectMultiScale(
    gray,
    scaleFactor=1.5,
    minNeighbors=6,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
  )

  for (x, y, width, height) in faces:
    cv2.rectangle(frame, (x,y), (x+width, y+height), (255,255,0), 2)

  cv2.imshow("faces", frame)
  if cv2.waitKey(1) == ord("q"):
    break
  
camera.release()
cv2.destroyAllWindows()