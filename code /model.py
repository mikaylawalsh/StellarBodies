import torch
from torch import nn

class StellarCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        self.num_epoch = 50
        self.batch_size = 32
        self.num_classes = num_classes
        self.lr = 0.01
        self.hidden_size1 = 64
        self.hidden_size2 = 128
        self.kernel_size = 3 

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.lr)

        self.conv1 = torch.nn.Conv2d(in_channels=input_channels, out_channels=self.hidden_size1, kernel_size=self.kernel_size, stride=1)
        self.leaky1 = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=self.hidden_size1, out_channels=self.hidden_size2, kernel_size=self.kernel_size, stride=1)
        self.leaky2 = torch.nn.LeakyReLU()
        self.flat = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(in_features=self.hidden_size2, out_features=self.num_classes)
        self.softmax = torch.nn.Softmax()
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out_conv1 = self.conv1(X)
        out_leaky1 = self.leaky1(out_conv1)
        out_conv2 = self.conv2(out_leaky1)
        out_leaky2 = self.leaky2(out_conv2)
        out_flat = self.flat(out_leaky2)
        out_dense1 = self.dense1(out_flat)
        logits = self.softmax(out_dense1)
        
        return logits

    def accuracy(self, logits: torch.Tensor, Y: torch.Tensor) -> float:
        preds = torch.argmax(logits, 1)
        num_correct = torch.sum(preds==Y)
        return num_correct/Y.shape[0]

def train(model: nn.Module, X: torch.Tensor, Y: torch.Tensor):
    model.train()
    for e in range(model.num_epoch):
        epoch_loss = 0
        epoch_acc = 0
        for b1 in range(model.batch_size, X.shape[0] + 1, model.batch_size):
            b0 = b1 - model.batch_size
            batchX = X[b0:b1]
            batchY = Y[b0:b1]

            logits = model(batchX)
            model.optimizer.zero_grad()
            loss = model.loss_fn(logits, batchY)
            loss.backward()
            model.optimizer.step()
            acc = model.accuracy(logits, batchY)
            epoch_loss += loss.item()
            epoch_acc += acc
        
        print("Epoch: ", e, "\t", "Loss: ", epoch_loss/model.batch_size, "\t", "Acc: ", epoch_acc/model.batch_size)
        

def test(model: nn.Module, X: torch.Tensor, Y: torch.Tensor):
    model.eval()

    with torch.no_grad():
        tot_loss = 0
        tot_acc = 0
        for b1 in range(model.batch_size, X.shape[0] + 1, model.batch_size):
            b0 = b1 - model.batch_size
            batchX = X[b0:b1]
            batchY = Y[b0:b1]

            logits = model(batchX)
            loss = model.loss_fn(logits, batchY)
            acc = model.accuracy(logits, batchY)
            tot_loss += loss.item()
            tot_acc += acc
        
        print("Test:\t", "Loss: ", tot_loss/model.batch_size, "\t", "Acc: ", tot_acc/model.batch_size)

