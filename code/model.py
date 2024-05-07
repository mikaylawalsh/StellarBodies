import torch
from torch import nn

class StellarCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(StellarCNN, self).__init__()
        self.num_epoch = 15
        self.batch_size = 32
        self.num_classes = num_classes
        self.lr = 0.0001
        self.hidden_size1 = 32
        self.hidden_size2 = 16
        self.kernel_size = 9

        self.loss_fn = nn.CrossEntropyLoss()

        self.conv1 = torch.nn.Conv2d(in_channels=input_channels, out_channels=self.hidden_size1, kernel_size=self.kernel_size, stride=3)
        self.leaky1 = torch.nn.LeakyReLU()
        # add max pooling
        self.max_pool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(in_channels=self.hidden_size1, out_channels=self.hidden_size2, kernel_size=self.kernel_size, stride=3)
        self.leaky2 = torch.nn.LeakyReLU()
        self.max_pool2 = torch.nn.MaxPool2d(3)
        self.flat = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(in_features=864, out_features=520)
        self.leaky3 = torch.nn.LeakyReLU()
        self.dense2 = torch.nn.Linear(in_features=520, out_features=128)
        self.leaky4 = torch.nn.LeakyReLU()
        self.dense3 = torch.nn.Linear(in_features=128, out_features=self.num_classes)
        # self.dense1 = torch.nn.Linear(in_features=3291520, out_features=self.num_classes)
        self.softmax = torch.nn.Softmax()
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.type(torch.float32)
        out_conv1 = self.conv1(X)
        out_leaky1 = self.leaky1(out_conv1)
        out_max = self.max_pool1(out_leaky1)
        out_conv2 = self.conv2(out_max)
        out_leaky2 = self.leaky2(out_conv2)
        out_max2 = self.max_pool2(out_leaky2)
        out_flat = self.flat(out_max2)
        out_dense1 = self.dense1(out_flat)
        out_leaky3 = self.leaky3(out_dense1)
        out_dense2 = self.dense2(out_leaky3)
        out_leaky4 = self.leaky4(out_dense2)
        out_dense3 = self.dense3(out_leaky4)
        logits = self.softmax(out_dense3)
        
        return logits

    def accuracy(self, logits: torch.Tensor, Y: torch.Tensor) -> float:
        preds = torch.argmax(logits, 1)
        num_correct = torch.sum(preds==Y)
        return num_correct/Y.shape[0]

def train(model: nn.Module, X: torch.Tensor, Y: torch.Tensor):
    optimizer = torch.optim.Adam(model.parameters(), model.lr)
    model.train()
    for e in range(model.num_epoch):
        epoch_loss = 0
        epoch_acc = 0
        for b1 in range(model.batch_size, X.shape[0] + 1, model.batch_size):
            b0 = b1 - model.batch_size
            batchX = X[b0:b1]
            batchY = Y[b0:b1]

            logits = model(batchX)
            optimizer.zero_grad()
            # logits = logits.type(torch.LongTensor)
            batchY = batchY.type(torch.long)
            loss = model.loss_fn(logits, batchY)
            loss.backward()
            optimizer.step()
            acc = model.accuracy(logits, batchY)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        print("Epoch: ", e, "Loss: ", epoch_loss/(X.shape[0]/model.batch_size), "\t", "Acc: ", epoch_acc/(X.shape[0]/model.batch_size))
        

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
            batchY = batchY.type(torch.long)
            loss = model.loss_fn(logits, batchY)
            acc = model.accuracy(logits, batchY)
            tot_loss += loss.item()
            tot_acc += acc
        
        print("Test:\t", "Loss: ", tot_loss/(X.shape[0]/model.batch_size), "\t", "Acc: ", tot_acc/(X.shape[0]/model.batch_size))

