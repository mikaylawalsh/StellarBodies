import numpy as np
import torch
from model import StellarCNN, train, test

def main():
    
    # get data
    X = None
    Y = None

    model = StellarCNN(X.shape[0], 5)

    train(model, X, Y)
    test(model, X, Y)


if __name__ == "__main__":
    main()