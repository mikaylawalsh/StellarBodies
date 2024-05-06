import sys
import time
import cv2
import numpy as np
import torch

class CameraConnection(): 
    def __init__(self, camera_input: int, out_file: str): 
        super(CameraConnection, self).__init__()
        print("here")
        self.stream = cv2.VideoCapture(camera_input)
        self.model = torch.load('best_model')
        self.model.eval()
        self.out_file = out_file
        self.device = 'cpu'

    def downsample(picture):
        down = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.MaxPool2d(2, stride=2),
        )
        return down(picture)

    def score_frame(self, frame):
        self.model.to(self.device)
        frame_tens = [torch.tensor(frame)] # need list wrapper? 
        # add in downsampling (3 times) before passing in 
        frame_tens = self.downsample(frame_tens)
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
                        9: "ursaminor"}
            label = labels_dict[pred]
        # labels = results.xyxyn[0][:, -1].numpy()
        # coordinates = results.xyxyn[0][:, :-1].numpy()
        return label
    
    def run_camera(self):
        # make sure that it is there 
        assert self.stream.isOpened() 

        x_shape = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG") #idk wtf this is but internet said to do it 

        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape)) 

        frame_count = 0
        fps = 0
        tfc = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        tfcc = 0

        # first frame
        ret, frame = self.stream.read() 

        while ret: 
            frame_count += 1

            # start_time = time.time() 
            label = self.score_frame(frame) # score frame
            if label:
                img = cv2.putText(frame, label, (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2) 
                cv2.imShow('Constellation Viewer', img)
            # frame = self.plot_boxes(results, frame) # plot the box
            # end_time = time.time()
            # fps = 1/np.round(end_time - start_time, 3) # measure the FPS
            # print(f"Frames Per Second : {fps}")
            out.write(frame) # write the frame onto the output

            ret, frame = self.stream.read() # read next frame

if __name__ == "__main__":
    camera_input = sys.argv[1]
    output_file = sys.argv[2]
    camera = CameraConnection(camera_input, output_file)
    camera.run_camera()