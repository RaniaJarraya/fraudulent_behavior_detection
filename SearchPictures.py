# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 19:58:17 2018

@author: ASUS
"""

import numpy as np
from flask import Flask, request, render_template
from PIL import Image  
import PIL
import cv2
from math import sqrt
import sys
import os
 
def Sqr(a):
    return a*a



def predect(a):
    MODE = "COCO"
        
    if MODE is "COCO":
        protoFile = "C:/Users/Rania/Desktop/AIM2/ferd/pose/coco/pose_deploy_linevec.prototxt"
        weightsFile = "C:/Users/Rania/Desktop/AIM2/ferd/pose/coco/pose_iter_440000.caffemodel"
        nPoints = 18
        POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
        POSE_WARNING=[ [0,8,86.3481],[0,11,86.3481],[0,9,167.58579],[0,12,167.58579],[7,1,80.000],[4,1,80.000],[6,11,30.000],[6,17,86.000],[4,10,30.000],[2,1,20.000],[4,7,320.000],[10,13,250.000]]
       
        
    elif MODE is "MPI" :
        protoFile = "C:/Users/Rania/Desktop/AIM2/ferd/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "C:/Users/Rania/Desktop/AIM2/ferd/pose/mpi/pose_iter_160000.caffemodel"
        nPoints = 15
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
        
        
        
    frame = cv2.imread(a)
    name=a.split('.')[0]
    name2=name+"_New.jpg"
    print(name2)
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1    
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    
    net.setInput(inpBlob)
    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]
    
    
    points = []
    
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
    
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
    
        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            print(int(x), int(y))
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    print(points)
    warn_dist=[]
    for pair in POSE_WARNING:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB] :
            if partA==10 and partB==13 :
                d=sqrt(Sqr(points[10][1]-points[13][1])+Sqr(points[10][0]-points[13][0]))
                if d>250.000:
                    warn_dist.append(d)
                    print(d,partA,partB)
            elif partA==4 and partB==7 :
                d=sqrt(Sqr(points[4][1]-points[7][1])+Sqr(points[4][0]-points[7][0]))
                if d>300.000:
                    warn_dist.append(d)
                    print(d,partA,partB)
            
            else:
                d=sqrt(Sqr(points[partA][1]-points[partB][1])+Sqr(points[partA][0]-points[partB][0]))
                if (d<pair[2]):
                    warn_dist.append([d,partA,partB])
                    print(d,partA,partB)
  
    if len(warn_dist) :
        cv2.putText(frameCopy, "warning detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            
    cv2.imshow('result', frameCopy)
   
    #cv2.imshow('Output-Skeleton', frame)  
    cv2.imwrite(name2, frameCopy)
    ret=[name2,warn_dist ]

    return ret    


app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "C:/Users/Rania/Desktop/AIM2/ferd/data_pose/1/036308230/"
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        result_img_path=""
        file = request.files['query_img']
        uploaded_img_path = "static/" + file.filename
        print(uploaded_img_path)
        pred=predect(uploaded_img_path)
        print(pred)
        result_img_path=pred[0]
        inter=pred[1]
        message=""
        if len(inter)==0:
            message="no warning case"
        else:
            message ="warning case " 
            
        print(result_img_path)
        return render_template('index.html',query_path=uploaded_img_path,answers=result_img_path,message=message)
    else:
        return render_template('index.html')


if __name__=="__main__":
     app.run(debug=True)
      
    