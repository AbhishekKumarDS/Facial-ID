import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import sys
# extract a single face from a given photograph

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def extract_face(filename,detector=detector,required_size=(224, 224)):
	pixels = plt.imread(filename)
	# create the detector, using default weights
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
    # resize pixels to the model size
    
def get_embeddings(face,model=model):
	# extract faces
    
	# convert into an array of samples
	samples = asarray(face, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	samples = samples[np.newaxis,:]
	# create a vggface model
	#model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat
# define a video capture object 

args=list(sys.argv)
idfile=args[-1]
ID_face = extract_face(idfile,detector)
ID_embedding=get_embeddings(ID_face,model)


cap = cv2.VideoCapture(0)
flag=False
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    #Use MTCNN to detect faces
    try:
        result = detector.detect_faces(frame)
        if result != []:
            for person in result:
                x1, y1, width, height = result[0]['box']
                x2, y2 = x1 + width, y1 + height
                # extract the face
                face = frame[y1:y2, x1:x2]
                # resize pixels to the model size
                subject_face = Image.fromarray(face)
                required_size=(224, 224)
                subject_face = subject_face.resize(required_size)
                sample = asarray(subject_face, 'float32')
                sample = preprocess_input(sample, version=2)
                subject_embeddings = get_embeddings(subject_face)
                score = cosine(ID_embedding, subject_embeddings)
                thresh = 0.5
                if score <= thresh:
                    print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
                    flag=True
                else:
                    print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
    except ValueError:
        pass
        '''cv2.rectangle(frame,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0,155,255),
                      2)

        cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
    #display resulting frame
        cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)'''
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & (flag or 0xFF == ord('q')):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()