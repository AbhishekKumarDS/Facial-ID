# Facial-ID
Face Identification with MTCNN and VGGFace2

To install and setup the dependencies, just run “Installer.bat” file. Internet Connection is required and around 400 Mb of Packages will be downloaded. 


Run “faceDIV.bat” to open cmd, initialise the environment and wait for further commands. 


Put any photos to be inferred inside the root directory of this project. 


To detect, crop and display a face from a photo (say, eg. a PAN card named ‘PAN.jpg’), 
type “ python face_detect.py PAN.jpg ”.


To check if 2 given photos are of the same person (say, ‘pic1.jpg’ and ‘pic2.jpg’), 
type “ python face_match.py pic1.jpg pic2.jpg ”.


To verify that a person in front of the camera is the same as verification photo provided by the person (say “ID.jpg”), 
type “ python face_verify.py ID.jpg ”.
This will start the webcam, check if the person in the feed is the same as in ID.jpg, and close the program when the person is verified to be the same. This can be used for Aadhar/PAN card verification, or for security/attendance purposes.
