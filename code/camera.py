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

    def score_frame(self, frame):
        self.model.to(self.device)
        frame_tens = [torch.tensor(frame)] # need list wrapper? 
        # add in downsampling (3 times) before passing in 
        results = self.model(frame_tens)
        pred = torch.argmax(results, 1)
        # find label from pred using dict 
        labels_dict = {}
        label = labels_dict[pred]
        # labels = results.xyxyn[0][:, -1].numpy()
        # coordinates = results.xyxyn[0][:, :-1].numpy()
        return label

    def show_boxes(self, results, frame): 
        labels, cords = results
        n_labels = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        classes = self.model.names 

        for i in range(n_labels):
            row = cords[i]

            # if score is less than a threshold we avoid making a prediction
            if row[4] < 0.2: 
                continue

            x1 = int(row[0]*x_shape)
            y1 = int(row[1]*y_shape)
            x2 = int(row[2]*x_shape)
            y2 = int(row[3]*y_shape)
            box_color = (0, 255, 0) # color of the box, currently lime green

            label_font = cv2.FONT_HERSHEY_SIMPLEX # default font

            # plot box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2) 
            
            # put text on box
            cv2.putText(frame, classes[labels[i]],(x1, y1), label_font, \
                        0.9, box_color, 2) 
            
        return frame
    
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

            start_time = time.time() 
            results = self.score_frame(frame) # score frame
            frame = self.plot_boxes(results, frame) # plot the box
            end_time = time.time()
            fps = 1/np.round(end_time - start_time, 3) # measure the FPS
            # print(f"Frames Per Second : {fps}")
            out.write(frame) # write the frame onto the output

            ret, frame = self.stream.read() # read next frame

if __name__ == "__main__":
    camera_input = sys.argv[1]
    output_file = sys.argv[2]
    camera = CameraConnection(camera_input, output_file)
    camera.run_camera()