import torch
from torch import nn

class VisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformer, self).__init__()
        self.flat = nn.Flatten()
        self.input_embedding = nn.Linear(840*560*3, 512) 
        self.transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.pos_encoder = nn.Parameter(torch.randn(1, 32, 512))
        self.max_pool = torch.nn.MaxPool2d(3)
        self.fc = nn.Sequential(
            nn.Linear(1700, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_classes)
            )
        # self.softmax = torch.nn.Softmax()

        self.loss_fn = nn.CrossEntropyLoss()

        self.num_epoch = 30
        self.batch_size = 32
        self.lr = 0.0001
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = torch.reshape(x, (32, -1, 512))
        x = x.float()
        x = self.input_embedding(x)
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = self.max_pool(x)
        x = self.flat(x)
        x = self.fc(x)
        # x = self.softmax(x)
        return x
    
    def accuracy(self, logits: torch.Tensor, Y: torch.Tensor) -> float:
        preds = torch.argmax(logits, 1)
        num_correct = torch.sum(preds==Y)
        return num_correct/Y.shape[0]