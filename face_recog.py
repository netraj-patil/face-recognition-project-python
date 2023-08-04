import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# load known faces

#itachi_image = face_recognition.load_image_file("faces/itachi.jpg")
#itachi_encoding = face_recognition.face_encodings(itachi_image)[0]

madara_image = face_recognition.load_image_file("faces/madara.jpg")
madara_encoding = face_recognition.face_encodings(madara_image)[0]

jiraiya_image = face_recognition.load_image_file("faces/jiraiya.jpg")
jiraiya_encoding = face_recognition.face_encodings(jiraiya_image)[0]

encodings = [madara_encoding, jiraiya_encoding]
names = ["Madara", "Jiraiya"]

#list of expected student
students = ["Madara", "Jiraiya"]

face_locations = []
face_encodings = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv","w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #recognise face
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(encodings, face_encoding)
        face_distance = face_recognition.face_distance(encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = names[best_match_index]

        #add text
            if name in names:
                font = cv2.FONT_HERSHEY_COMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale = 1.5
                fontColor = (255,0,0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name,bottomLeftCornerOfText,font,fontScale, fontColor, thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow(name)

        cv2.imshow("Attendence", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()