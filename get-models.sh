#!/usr/bin/bash 

mkdir -p models
wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 -P models
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P models

bzip2 -d models/dlib_face_recognition_resnet_model_v1.dat.bz2
bzip2 -d models/shape_predictor_68_face_landmarks.dat.bz2 

rm -rf models/dlib_face_recognition_resnet_model_v1.dat.bz2
rm -rf models/shape_predictor_68_face_landmarks.dat.bz2 