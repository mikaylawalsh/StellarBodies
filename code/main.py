import numpy as np
import torch
from model import StellarCNN, train, test
from vis_trans import VisTrans
from ViT import VisionTransformer
import cv2 
import torchvision.transforms as transforms

def main():
    
    X1 = torch.load("data/images.pt")
    Y1 = torch.load("data/labels.pt")
    Y1 = torch.squeeze(Y1)

    X2 = torch.load("data/images10.pt")
    Y2 = torch.load("data/labels10.pt")
    Y2 = torch.squeeze(Y2)

    X = torch.cat((X1, X2), 0)
    Y = torch.cat((Y1, Y2), 0)

    indices = torch.randperm(X.shape[0])
    X = X[indices]
    Y = Y[indices]

    num_classes = len(torch.unique(Y))
    # model = VisTrans((X.shape[2], X.shape[3]), num_classes)
    model = VisionTransformer(num_classes)
    # model = StellarCNN(3, 10)

    split = int(len(X)*.8)

    train(model, X[:split], Y[:split])
    test(model, X[split:], Y[split:])

    # model.load_state_dict(torch.load('../model_state2.pt'))
    model.eval()
    preds = []
    for im in range(1, 11):
        image = cv2.imread(f'live_data/live{im}.JPG')
  
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
        transform = transforms.Compose([ 
            transforms.ToTensor() 
        ])
    
        tensor = transform(image)

        frame_tens = transforms.Resize(size=(560,840))(tensor)
        frame_tens = frame_tens.repeat(32, 1, 1, 1)

        results = model(frame_tens)

        pred = torch.argmax(results, 1)[0].item()
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

        preds.append(label)
    
    print(preds)

    # torch.save(model.state_dict(), '../model_state')

if __name__ == "__main__":
    main()