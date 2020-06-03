import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#Fun to detect face...

def detect_face(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_cas = cv2.CascadeClassifier('/home/prathamesh/Documents/projects/face_recognizition/HaarCascade/haar.xml')
    faces = face_cas.detectMultiScale(gray,scaleFactor = 1.3,minNeighbors=4);
    
    if(len(faces)==0):
        return None,None
    
    
    faces[0] = (x,y,w,h)
    return gray[y:y+w ,x:x+h] ,faces[0]

#fun to read training images and detect faces

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
        
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
        
        image_path = subject_dir_path + "/" + image_name
        image = cv2.imread(image_path)
        resized_img = cv2.resize(image,(1000,700))
        
        cv2.imshow("Training on image...", resized_img)
        cv2.waitKey(100)
        
        
        face,rect = detect_face(resized_img)
        if face is not None:
            faces.append(face)
            labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels
        
    
    
print("Preparing data...")
faces, labels = prepare_training_data('/home/prathamesh/Documents/projects/face_recognizition/strainingimages')
print("Data prepared")
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

#create our LBPH face recognizer
face_recognizer = cv2.face.createLBPHFaceRecognizer()
#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        
        
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    label= face_recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img

test_img1 = cv2.imread("/home/prathamesh/Documents/projects/face_recognizition/Testimages/prathamesh.jpg")
test_img2 = cv2.imread("/home/prathamesh/Documents/projects/face_recognizition/Testimages/p5.jpg")

predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")


#create a figure of 2 plots (one for each test image)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#display test image1 result
ax1.imshow(cv2.cvtColor(predicted_img1, cv2.COLOR_BGR2RGB))
#display test image2 result
ax2.imshow(cv2.cvtColor(predicted_img2, cv2.COLOR_BGR2RGB))
#display both images
cv2.imshow("pratham test", predicted_img1)
cv2.imshow("pratham test", predicted_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
