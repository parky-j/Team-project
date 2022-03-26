#!/usr/bin/env python
# coding: utf-8

# In[16]:


import cv2
import mediapipe as mp
import math
import numpy as np

class poseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1, save_route=None):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.save_route = save_route
    
    
    def findAngle3D(self, p1, p2, p3):
        ba = p1 - p2
        bc = p3 - p2

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        if np.degrees(angle) > 180 :
            angle = 360.0 - np.degrees(angle)
        else:
            angle = np.degrees(angle)

        return angle
    
    def findAngle3D2(self, p1, p2, p3):
        ba = p1 - p2
        bc = p3 - p2

        v = np.array([ba,bc])
        v=v/np.linalg.norm(v, axis=1)[:, np.newaxis]
        angle=np.arccos(np.einsum('nt,nt->n',
                         v[[0],:],
                         v[[1],:]))

        angle = np.degrees(angle)
        
        return angle    
    
    def findLandmarks(self, cap):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils 
        mp_drawing_styles = mp.solutions.drawing_styles
        
        w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        delay = round(1000/fps)
        out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

        rightLegAngleList = []
        leftLegAngleList = []
        rightArmAngleList = []
        leftArmAngleList = []
        lmlist = []
        flag = 0
        
        with mp_pose.Pose(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence, model_complexity=self.model_complexity) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # RGB to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                except:
                    pass

                # Save landmarks
                if results.pose_landmarks is not None:
                    mypose = results.pose_landmarks
                    tmpList = []
                    for idx,lm in enumerate(mypose.landmark):
                        point=np.array([int(idx), lm.x, lm.y, lm.z])
                        tmpList.append(point)
                    lmlist.append(tmpList)

                    # Calc Leg Angles
                    rightLegAngleList.append(self.findAngle3D(tmpList[24][1:],tmpList[26][1:],tmpList[28][1:]))
                    leftLegAngleList.append(self.findAngle3D(tmpList[23][1:],tmpList[25][1:],tmpList[27][1:]))
                    rightArmAngleList.append(self.findAngle3D(tmpList[12][1:],tmpList[14][1:],tmpList[16][1:]))
                    leftArmAngleList.append(self.findAngle3D(tmpList[11][1:],tmpList[13][1:],tmpList[15][1:]))

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                         )  
                if(self.save_route != None):
                    cv2.imwrite(self.save_route % (flag), image)
                    flag = flag + 1

                out.write(image)

                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            out.release()
            cap.release()
            cv2.destroyAllWindows()
            
            return lmlist, rightLegAngleList, leftLegAngleList, rightArmAngleList, leftArmAngleList