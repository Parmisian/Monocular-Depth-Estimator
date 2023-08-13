# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 17:07:29 2023

@author: ulas0
"""

import torch
import cv2
import matplotlib.pyplot as plt


midas= torch.hub.load('intel-isl/MiDaS','DPT_Hybrid')
midas.to('cpu')
midas.eval()

transform= torch.hub.load('intel-isl/MiDaS', 'transforms')

transform = transform.dpt_transform


cap= cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame= cap.read()
    
    img= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')
    
    with torch.no_grad():
        prediction= midas(imgbatch)
        prediction= torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size= img.shape[:2],
                mode='bicubic',
                align_corners=False
                ).squeeze()
        
        #prediction= prediction.clamp(min=0, max=255)
        
        output =prediction.cpu().numpy()
        
        plt.imshow(output)
        cv2.imshow('CV2Frame', frame)
        plt.pause(0.0001)
            
        print("Pred Value: ",prediction)
    cv2.imshow('CV2Frame', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()        
plt.show()

    
