import cv2
from simple_facerec import SimpleFacerec

# Initialiser SimpleFacerec
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")  # Remplacez par le chemin du dossier contenant vos images

# Capturer la vidéo
cap = cv2.VideoCapture(0)  # Remplacez 0 par l'index correct si nécessaire

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur : Impossible de lire le flux vidéo.")
        break

    # Reconnaissance faciale
    face_locations, face_names = sfr.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_names):
        top, right, bottom, left = face_loc
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher la vidéo
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Touche ESC pour quitter
        break

cap.release()
cv2.destroyAllWindows()
