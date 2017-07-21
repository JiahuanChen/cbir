# draw the VLAD of the first 11 images with 16 clusters, like the paper did...
# each row represent an image
import numpy as np
import cv2

def draw_small(col,row,des):
    m = 0.025
    p1 = (col+5,row+5)
    for i in range(8):
        mag = abs(des[i])
        p2 = (int(round(col+5 + mag*np.cos(np.pi/4*i)/m )),
              int(round(row+5 - mag*np.sin(np.pi/4*i)/m ))
              )
        if des[i]<0:
            cv2.arrowedLine(img, p1, p2, (0,0,255), 1)
        else:
            cv2.arrowedLine(img, p1, p2, (0,255,0), 1)

def draw_big(col,row,des):
    o_col = 0
    o_row = 0
    for index in range(0,128,8):
        print(o_col,o_row)
        draw_small(col + o_col*10,row+o_row*10,des[index:index+8])
        o_col += 1
        if o_col >=4:
            o_col = 0
            o_row += 1
        if o_row >= 4:
            break

img = np.zeros((600,900,3), np.uint8)
vlad = np.load('vlad.pickle')
vlad = vlad.reshape(-1,128)

counter = 0
col = 0
row = 0
for des in vlad:
    draw_big(5+col*50,5+row*50,des)
    col += 1
    if col >=16:
        col = 0
        row += 1
    if row > 10:
        break

cv2.imshow('VLAD', img)

cv2.waitKey(0)
