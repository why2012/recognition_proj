# coding:utf-8
import cv2
import numpy as np

def BGR2HSV(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def HSV2BGR(img):
	return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def getImgSize(img):
	return (img.shape[1], img.shape[0])

def createWhiteColorImg(size):
	return np.uint8(np.ones((size[1], size[0], 3)) * 255)

def filterBlack(img, color1 = [0, 0, 0], color2 = [180, 255, 90]):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_black = np.array(color1)
    upper_black = np.array(color2)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    img = createWhiteColorImg(getImgSize(img))
    img = cv2.bitwise_and(img, img, mask = mask)
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
    return img

def filterBlackOriginImg(img, color1 = [0, 0, 0], color2 = [180, 255, 90], thresh1 = 10, thresh2 = 255):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_black = np.array(color1)
    upper_black = np.array(color2)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    img = cv2.bitwise_and(img, img, mask = mask)
    img[img[:, :, 0: 3] == [0, 0, 0]] = 255
    return img

def filterBlue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_black = np.array([100, 50, 50])
    upper_black = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    img = createWhiteColorImg(getImgSize(img))
    img = cv2.bitwise_and(img, img, mask = mask)
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
    return img

def videoCap():
	cap = cv2.VideoCapture(0)
	while(1):
	    # Take each frame
	    _, frame = cap.read()
	    # Convert BGR to HSV
	    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	    # define range of blue color in HSV
	    lower_blue = np.array([100,30,30])
	    upper_blue = np.array([130,255,255])
	    # Threshold the HSV image to get only blue colors
	    mask = cv2.inRange(hsv, lower_blue, upper_blue)
	    # Bitwise-AND mask and original image
	    res = cv2.bitwise_and(frame, frame, mask = mask)
	    cv2.imshow('frame',frame)
	    cv2.imshow('mask',mask)
	    cv2.imshow('res',res)
	    k = cv2.waitKey(5) & 0xFF
	    if k == 27:
	        break
	cv2.destroyAllWindows()
	cap.release()

def showImgs(*imgs):
	index = 0
	for img in imgs:
		cv2.imshow("img" + str(index), img)
		index += 1
	cv2.waitKey(10)

