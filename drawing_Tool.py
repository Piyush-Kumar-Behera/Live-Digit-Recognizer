import cv2
import numpy as np

x_prev,y_prev = 0,0
drawing = False
image = np.ones((512,512,3), np.uint8) * 150
x_max,x_min,y_max,y_min = 0,1000,0,1000

def click_event(event, x, y, flags, param):
    global x_prev,y_prev,drawing,image,x_max,x_min,y_max,y_min
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_prev,y_prev = x,y
        x_max = max(x_max, x_prev)
        x_min = min(x_min, x_prev)

        y_max = max(y_max, y_prev)
        y_min = min(y_min, y_prev)

    elif event == cv2.EVENT_MOUSEMOVE and drawing == True:
        cv2.line(image, (x_prev,y_prev), (x,y), (0,255,255), 10)
        x_prev,y_prev = x,y
        x_max = max(x_max, x_prev)
        x_min = min(x_min, x_prev)

        y_max = max(y_max, y_prev)
        y_min = min(y_min, y_prev)

        cv2.imshow('image', image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(image, (x_prev,y_prev), (x,y), (0,255,255), 10)
        cv2.imshow('image', image)
        x_prev,y_prev = x,y
        x_max = max(x_max, x_prev)
        x_min = min(x_min, x_prev)

        y_max = max(y_max, y_prev)
        y_min = min(y_min, y_prev)


cv2.imshow('image', image)

cv2.setMouseCallback('image', click_event)


cv2.waitKey(0)

cropped_image = image[y_min : y_max, x_min : x_max, : ]
cv2.imshow("Cropped",cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
