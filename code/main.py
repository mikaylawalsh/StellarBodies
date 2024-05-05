import numpy as np
import torch
from model import StellarCNN, train, test

def main():
    
    X = torch.load("data/images.pt")
    Y = torch.load("data/labels.pt")
    Y = torch.squeeze(Y)

    indices = torch.randperm(X.shape[0])
    X = X[indices]
    Y = Y[indices]

    num_classes = len(torch.unique(Y))

    model = StellarCNN(X.shape[1], num_classes)

    split = int(len(X)*.8)

    train(model, X[:split], Y[:split])
    test(model, X[split:], Y[split:])

if __name__ == "__main__":
    main()