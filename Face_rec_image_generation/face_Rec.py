import cv2
import face_recognition
import numpy as np

# Load a sample picture and learn how to recognize it.
my_image = face_recognition.load_image_file("ME/me.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]

# Load a second sample picture and learn how to recognize it.
ronaldo_image = face_recognition.load_image_file("Ronaldo/ronaldo.webp")
ronaldo_face_encoding = face_recognition.face_encodings(ronaldo_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    my_face_encoding,
    ronaldo_face_encoding,
]
known_face_names = [
    "Pranav",
    "Ronaldo",
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []


test_image = face_recognition.load_image_file("test_images/ronaldo2.jpg")



# For eye and Face detection                  
detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
faces=detector.detectMultiScale(test_image,1.1,7)
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
 #Draw the rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(test_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = test_image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)



# Resize frame of video to 1/4 size for faster face recognition processing
small_frame = cv2.resize(test_image, (0, 0), fx=0.25, fy=0.25)
# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#rgb_small_frame = small_frame[:, :, ::-1] #old code,does not work 
rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) 

# Find all the faces and face encodings in the current frame of video
face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
face_names = []
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)


# Display the results
print(name)

# Making an h5 file to use it on our Flask App


for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    # Draw a box around the face
    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(test_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(test_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
            

test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)


_, test_image = cv2.imencode('.jpg', test_image)
# as the cv2.imencode function returns a tuple with two values.


img_path = "uploads/modified_image.jpg"

cv2.imwrite(img_path, test_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# Read the image
original_image = cv2.imread("test_images/ronaldo2.jpg")

# Process the image

# Save the image without color space conversion
cv2.imwrite("uploads/modified_image.jpg", original_image)

with open(img_path, "wb") as img_file:
    img_file.write(test_image)


        
            

            





    

