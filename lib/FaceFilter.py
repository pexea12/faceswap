import dlib
import numpy as np
import cv2
import face_recognition
# import face_recognition_models

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
# facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

class FaceFilter():
    def __init__(self, reference_file_path, threshold = 0.6):
        # image = cv2.imread(reference_file_path)
        # rect = detector(image, 1)[0]
        # shape = predictor(image, rect)
        # self.encoding = np.array(facerec.compute_face_descriptor(image, shape))
        self.encoding = face_recognition.face_encodings(image)[0] 
        # Note: we take only first face, so the reference file should only contain one face. We could also keep all faces found and filter against multiple faces
        self.threshold = threshold
    
    def check(self, detected_face):
        # rect = detector(detected_face.image, 1)[0]
        # shape = predictor(detected_face.image, rect)
        # encodings = facerec.compute_face_descriptor(detected_face.image, shape)
        encodings = face_recognition.face_encodings(detected_face.image)[0] 
        # we could use detected landmarks, but I did not manage to do so
        # score = np.linalg.norm(self.encoding - encodings)
        score = face_recognition.face_distance([self.encoding], encodings)
        print(score)
        return score <= self.threshold


# # Copy/Paste (mostly) from private method in face_recognition
# face_recognition_model = face_recognition_models.face_recognition_model_location()
# face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# def convert(detected_face):
#     return np.array(face_encoder.compute_face_descriptor(detected_face.image, detected_face.landmarks, 1))
# # end of Copy/Paste
