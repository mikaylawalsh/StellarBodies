import sys
import time
import cv2
import numpy as np
import torch
import torchvision
from model import StellarCNN, train, test

class CameraConnection(): 
    def __init__(self): 
        super(CameraConnection, self).__init__()
        self.stream = cv2.VideoCapture(0)
        self.model = torch.load('../extra_model')
        self.model.eval()

    # def downsample(picture):
    #     down = torch.nn.Sequential(
    #         torch.nn.MaxPool2d(2, stride=2),
    #         torch.nn.MaxPool2d(2, stride=2),
    #         torch.nn.MaxPool2d(2, stride=2),
    #     )
    #     return down(picture)

    def score_frame(self, frame):
        self.model.to(self.device)
        frame_tens = [torch.tensor(frame)] # need list wrapper? 
        # add in downsampling (3 times) before passing in 
        # frame_tens = self.downsample(frame_tens)
        frame_tens = frame_tens[..., 100:1180]
        frame_tens = torchvision.transforms.Resize(size=(560,840))(frame_tens)
        results = self.model(frame_tens)
        label = None
        if torch.max(results) >= 0.5:
            pred = torch.argmax(results, 1)
            # find label from pred using dict 
            labels_dict = {0: "aquila",
                        1: "canismajor",
                        2: "cassiopeia",
                        3: "gemini",
                        4: "lyra",
                        5: "orion",
                        6: "pegasus",
                        7: "taurus",
                        8: "ursamajor",
                        9: "ursaminor",
                        10: "leo"}
            label = labels_dict[pred]
        return label
    
    def run_camera(self):
        # make sure that it is there 
        assert self.stream.isOpened() 

        # x_shape = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        # y_shape = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # four_cc = cv2.VideoWriter_fourcc(*"MJPG") #idk wtf this is but internet said to do it 

        # out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape)) 

        # frame_count = 0
        # fps = 0
        # tfc = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        # tfcc = 0

        # first frame
        ret, frame = self.stream.read() 


        while ret: 
            label = self.score_frame(frame) # score frame
            if label:
                img = cv2.putText(frame, label, (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.imShow('Constellation Viewer', img)
            else :
                cv2.imShow('Constellation Viewer', frame)

            cv2.waitKey()
            ret, frame = self.stream.read() # read next frame

if __name__ == "__main__":
    camera = CameraConnection()
    camera.run_camera()