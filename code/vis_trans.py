import torch
from torch import nn
import torch.nn.functional as F

class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch_size, channels, height, width = images.size()
        num_patches_h = height // self.patch_size[0]
        num_patches_w = width // self.patch_size[1]
        patches = images.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1])
        patches = torch.reshape(patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size[0] * self.patch_size[1] * channels,
            ))
        # patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size * self.patch_size).permute(0, 2, 3, 1)
        return patches

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(2352, projection_dim)
        self.leaky = nn.LeakyReLU()
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, patch):
        positions = torch.arange(start=0, end=self.num_patches, step=1).unsqueeze(0)
        patch = patch.float()
        projected_patches = self.projection(patch)
        projected_patches = self.leaky(projected_patches)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

class TransformerLayer(nn.Module):
    def __init__(self, projection_dim, num_heads):
        super().__init__()
        self.projection_dim = projection_dim
        self.layer_norm1 = nn.LayerNorm(projection_dim)
        self.layer_norm2 = nn.LayerNorm(projection_dim)
        self.multihead_att = nn.MultiheadAttention(projection_dim, num_heads, dropout=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(64, projection_dim*2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim*2, projection_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1) 
        )

    def forward(self, encoded_patches):
        x1 = self.layer_norm1(encoded_patches)
        attention_output, _ = self.multihead_att(x1, x1, x1)
        x2 = attention_output + encoded_patches
        x3 = self.layer_norm2(x2)
        # x3 = self.mlp(x3)
        output = x3 + x2

        return output
    
class VisTrans(nn.Module):
    def __init__(self, image_size, num_classes):
        super(VisTrans, self).__init__()

        self.patch_size = (28, 28)
        num_patches = (image_size[0] // self.patch_size[0]) * (image_size[1] // self.patch_size[1])
        projection_dim = 64
        num_heads = 4

        self.patches = Patches(self.patch_size)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        self.trans = TransformerLayer(projection_dim, num_heads)
        self.layer_norm = nn.LayerNorm(projection_dim) #?
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.mlp = nn.Sequential(
            nn.Linear(38400, 15000),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(15000, 920),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(920, projection_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(projection_dim, num_classes)
        self.softmax = torch.nn.Softmax()
        self.loss_fn = nn.CrossEntropyLoss()

        self.num_epoch = 10
        self.batch_size = 256
        self.lr = 0.001

    def forward(self, X):
        patches = self.patches(X)
        encoded_patches = self.patch_encoder(patches)
        trans = self.trans(encoded_patches)
        lay = self.layer_norm(trans)
        flat = self.flat(lay)
        drop = self.drop(flat)
        mlp = self.mlp(drop)
        clas = self.classifier(mlp)
        logits = self.softmax(clas)
        return logits  
    
    def accuracy(self, logits: torch.Tensor, Y: torch.Tensor) -> float:
        preds = torch.argmax(logits, 1)
        # print("PREDS: ", preds)
        num_correct = torch.sum(preds==Y)
        return num_correct/Y.shape[0]       