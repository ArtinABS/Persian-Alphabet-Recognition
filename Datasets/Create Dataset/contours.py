import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def bounding_rect(contours):
    x, y, w, h = cv.boundingRect(contours)
    return int(x), int(y), int(w), int(h)


def draw_bounding_rect(image, contours):
    x, y, w, h = cv.boundingRect(contours)
    return cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

# Area of Contours
def area(contours):
    return cv.contourArea(contours)

def perimeter(contours):
    return cv.arcLength(contours, True)

img = cv.imread("ALPHA.jpg")
result = img.copy()

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

alph_list = []

for cnt in contours:
    if area(cnt) > 200:
        # cv.drawContours(result, cnt, -1, (0, 255, 0), 1)
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print(area(cnt))
        alpha = img[y:y + h, x:x + w]
        alpha = cv.resize(alpha, (128,128))
        alph_list.append(alpha)
        cv.imshow("Live", result)
        # cv.imshow("Alphabet", alpha)
        cv.waitKey(0)
    else:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)


print(contours)
