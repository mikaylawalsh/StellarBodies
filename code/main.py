import numpy as np
import torch
from model import StellarCNN, train, test

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

    model = StellarCNN(X.shape[1], num_classes)

    split = int(len(X)*.8)

    train(model, X[:split], Y[:split])
    test(model, X[split:], Y[split:])

    torch.save(model, '../model')

if __name__ == "__main__":
    main()