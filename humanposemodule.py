import cv2
import mediapipe as mp
import time
import math

class poseDetector:
    def __init__(self, static_mode=False, upbbody=False, smooth=True, detection_confident= 0.5, tracking_confident=0.5):
        self.static_mode = static_mode
        self.upbbody = upbbody
        self.smooth = smooth
        self.detection_confident = detection_confident
        self.tracking_confident = tracking_confident
        
        self.mppose = mp.solutions.pose
        self.pose = self.mppose.Pose(self.static_mode, self.upbbody, self.smooth, self.detection_confident, self.tracking_confident) 
        self.mpdraw = mp.solutions.drawing_utils
        
        
    def findpose(self, frame, draw_landmark=True):
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img)
        
        if self.results.pose_landmarks:
            if draw_landmark:
                self.mpdraw.draw_landmarks(frame, self.results.pose_landmarks, self.mppose.POSE_CONNECTIONS)
        
        return frame
    
    def findlocation(self, frame, draw_landmark=True):
        self.lmlist = []
        if self.results.pose_landmarks:
            mypose = self.results.pose_landmarks
            
            for idx,lm in enumerate(mypose.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                
                self.lmlist.append([idx,cx,cy])
                if draw_landmark:
                    cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

        return self.lmlist
    
    def findAngle3D(self, p1, p2, p3):
        ba = p1 - p2
        bc = p3 - p2
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
