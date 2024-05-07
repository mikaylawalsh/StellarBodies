import numpy as np
import torch
from model import StellarCNN, train, test

model = torch.load('model')
torch.save(model.state_dict(), '../model_state2.pt')