#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pylab as plt
import cv2
import numpy as np
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask) 
    return masked_image
def process(image):
    print(image.shape)
    cv2.line(image,(0,500),(500,500),(0,255,0),thickness=1)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, 2*height/3),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = canny_image
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=200,
                            lines=np.array([]),
                            minLineLength=80,
                            maxLineGap=50)
    image_with_lines,dist = drow_the_lines(image, lines)
    print((lines))
    return image_with_lines
def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=1)
            distance=(y1+y2)/2-500
            print(f"distance : {distance}")

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img,distance
cap = cv2.VideoCapture('pass1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

