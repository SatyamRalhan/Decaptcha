from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import pickle
import imutils
from helpers import resize_to_fit


# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that 
# were given. The evaluation code may give unexpected results if this convention is not followed.

def decaptcha( filenames ):
	numChars=np.empty((len(filenames),))
	codes=[]
	MODEL_FILENAME = "captcha_model.hdf5"
	# MODEL_LABELS_FILENAME ="model_labels.dat"
	with open("model_labels.dat", "rb") as f:
		lb = pickle.load(f)
	model = load_model(MODEL_FILENAME)
	# captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
	# counts = {}
	for (i, captcha_image_file) in enumerate(filenames):
		image = cv2.imread(captcha_image_file)
		image2 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
		image2[:,:,0] = np.where(image2[:, :, 1] == 127, 0, image2[:, :, 0])
		image2[:,:,1] = np.where(image2[:, :, 1] == 127, 0, image2[:, :, 1])
		image2[:,:,2] = np.where(image2[:, :, 1] == 127, 0, image2[:, :, 2])

		image2[:,:,0] = np.where(np.logical_and(image2[:, :, 1] == 76, image2[:, :, 2] == 255), 179, image2[:, :, 0])
		image2[:,:,1] = np.where(np.logical_and(image2[:, :, 1] == 76, image2[:, :, 2] == 255), 255, image2[:, :, 1])
		image2[:,:,2] = np.where(np.logical_and(image2[:, :, 1] == 76, image2[:, :, 2] == 255), 255, image2[:, :, 2])

		cond = np.logical_and(np.logical_and(image2[:, :, 1] != 76, image2[:, :, 2] != 255), image2[:, :, 1] != 127)
		image2[:,:,0] = np.where(cond, 179, image2[:, :, 0])
		image2[:,:,1] = np.where(cond, 255, image2[:, :, 1])
		image2[:,:,2] = np.where(cond, 255, image2[:, :, 2])
		image = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = contours[1] if imutils.is_cv3() else contours[0]
		letter_image_regions = []

		for contour in contours:
			(x, y, w, h) = cv2.boundingRect(contour)
			if w<15 or h<50:
				continue
			else:
				letter_image_regions.append((x, y, w, h))

		letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
		output = cv2.merge([image] * 3)
		predictions = []

		for letter_bounding_box in letter_image_regions:
		# Grab the coordinates of the letter in the image
			x, y, w, h = letter_bounding_box
			letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
			letter_image = resize_to_fit(letter_image,20, 20)
			letter_image = np.expand_dims(letter_image, axis=2)
			letter_image = np.expand_dims(letter_image, axis=0)
			prediction = model.predict(letter_image)
			letter = lb.inverse_transform(prediction)[0]
			predictions.append(letter)
		captcha_text = "".join(predictions)
		numChars[i]=len(letter_image_regions)
		codes.append(captcha_text)
	return (numChars, codes)