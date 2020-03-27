#Importing Libraries and Classes

from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import tkinter.filedialog
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras import metrics
from keras.models import model_from_json

#Reading the trained model
json_file = open('model.json','r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Initializing global parameters
x_prev,y_prev = 0,0
drawing = False
draw_image = np.ones((550,550,3), np.uint8) * 0
line_color = (255,255,255)
panel_p1,panel_p2 = None, None
ans = -1
pen_size = 15


def check_range(img, w, h):
    #This function returns the the bounding box of the drawn image
    i = 0
    while sum(sum(img[:,i])) == 0:
        i = i + 1
    x_min = i - 1

    i = w - 1
    while sum(sum(img[:,i])) == 0:
        i = i - 1
    x_max = i + 1

    i = 0
    while sum(sum(img[i,:])) == 0:
        i = i + 1
    y_min = i - 1

    i = h - 1
    while sum(sum(img[i,:])) == 0:
        i = i - 1
    y_max = i + 1

    return x_max,x_min,y_max,y_min

def click_event(event, x, y, flags, param):
    #This is function for the click_event to draw
    global x_prev,y_prev,drawing,draw_image
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_prev,y_prev = x,y

    elif event == cv2.EVENT_MOUSEMOVE and drawing == True:
        cv2.line(draw_image, (x_prev,y_prev), (x,y), line_color, pen_size)
        x_prev,y_prev = x,y

        cv2.imshow('Drawing_Tool', draw_image)

    elif event == cv2.EVENT_LBUTTONUP and drawing == True:
        drawing = False
        cv2.line(draw_image, (x_prev,y_prev), (x,y), line_color, pen_size)
        cv2.imshow('Drawing_Tool', draw_image)
        x_prev,y_prev = x,y


root = Tk()
width = 1300
height = 800

def pad_with(vector, pad_width, iaxis, kwargs):
    #Funtion for padding the cropped image
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def B1_Drawing_tool():
    #Function for the draw button on tkinter app
    global draw_image, panel_p1, panel_p2
    draw_image = np.ones((550,550,3), np.uint8) * 0

    cv2.imshow('Drawing_Tool', draw_image)
    cv2.setMouseCallback('Drawing_Tool', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    draw_image_bw = cv2.cvtColor(draw_image, cv2.COLOR_BGR2GRAY)

    img_p4 = Image.fromarray(draw_image_bw)
    phtimg_p4 = ImageTk.PhotoImage(img_p4)
    panel_p1.configure(image = phtimg_p4)
    panel_p1.photo = phtimg_p4

def B2_recg_tool():
    #Funtion for Recognize button on the tkinter app
    global draw_image, panel_p1, panel_p2, x_max, x_min, y_max, y_min, ans


    x_max,x_min,y_max,y_min = check_range(draw_image, 550, 550)

    print('x-max = ' + str(x_max))
    print('x-min = ' + str(x_min))
    print('y-max = ' + str(y_max))
    print('y-min = ' + str(y_min))
    print(draw_image.shape)



    draw_image1 = draw_image[max(0,y_min-10) : min(y_max+10,549), max(0,x_min-10) :min(x_max+10,549), : ]

    draw_image_bw = cv2.cvtColor(draw_image1, cv2.COLOR_BGR2GRAY)
    draw_image_bw = np.pad(draw_image_bw,30, pad_with)
    pimg = cv2.resize(draw_image_bw, (28,28),interpolation = cv2.INTER_AREA)
    pimg = pimg/255

    pimg1 = pimg.reshape(1,28,28,1)

    y_pred = model.predict(pimg1)

    print(y_pred)
    ans = np.argmax(y_pred)

    load_num_img = cv2.imread(str(ans)+'.png',-1)
    load_num_img1 = cv2.resize(load_num_img, (550,550),interpolation = cv2.INTER_AREA)
    img_p5 = Image.fromarray(load_num_img1)
    phtimg_p5 = ImageTk.PhotoImage(img_p5)
    panel_p2.configure(image = phtimg_p5)
    panel_p2.photo = phtimg_p5


#To fix the dimension of the tkinter window
root.geometry(str(width)+'x'+str(height))

root.minsize(width,height)
root.maxsize(width,height)

#Designing and Labeling of the application
Head1 = Label(text = 'Live Digit Recognizer',font = ('Arial', 30))
Head1.pack(side = 'top')

Head2 = Label(text = 'Click on "Draw" to draw a new image of a digit, then press enter and click on "Recognize"',font = ('Helvetica', 15,'bold'))
Head2.pack(side = 'top')

Button1 = Button(text = 'Draw',font = ('Caveat',15,'italic'),fg = 'black',command = B1_Drawing_tool)
Button1.pack(side = 'top', pady = 10)

Button2 = Button(text = 'Recognize',font = ('Caveat',15,'italic'),fg = 'black', command = B2_recg_tool)
Button2.pack(side = 'top')


#Starting with blank images on the application
photo1 = np.ones((550,550), np.uint8) * 200

img_p1 = Image.fromarray(photo1)
phtimg_p1 = ImageTk.PhotoImage(img_p1)
panel_p1 = Label(image = phtimg_p1)
panel_p1.image = phtimg_p1
panel_p1.pack(side = 'left', padx = 30, pady = 10)


photo2 = np.ones((550,550), np.uint8) * 200

img_p2 = Image.fromarray(photo2)
phtimg_p2 = ImageTk.PhotoImage(img_p2)
panel_p2 = Label(image = phtimg_p2)
panel_p2.image = phtimg_p2
panel_p2.pack(side = 'right', padx = 30, pady = 10)

#Mainloop of the root
root.mainloop()
