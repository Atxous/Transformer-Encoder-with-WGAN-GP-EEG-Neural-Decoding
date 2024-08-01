import torch
import math

class GaussianLayer(torch.nn.Module):
    def __init__(self, input_dim, kernel_intializer = "xavier", bias_initializer = "zeros"):
        super(GaussianLayer, self).__init__()
        self.std = torch.nn.Parameter(torch.empty([1, input_dim]))
        self.mean = torch.nn.Parameter(torch.empty([1, input_dim]))
        
        if kernel_intializer == "xavier":
            torch.nn.init.xavier_uniform_(self.std)
        if bias_initializer == "zeros":
            torch.nn.init.zeros_(self.mean)
            
    def forward(self, x, noise):
        return x * (self.mean + self.std * noise)
    
    
class EEGPositionalEncoder(torch.nn.Module):
    def __init__(self, max_length, embed_size, embed_first = False):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        
        p = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pos_embed = torch.zeros(1, max_length, embed_size)
        pos_embed[0, :, ::2] = torch.sin(p * div_term)
        pos_embed[0, :, 1::2] = torch.cos(p * div_term)
        if embed_first:
            pos_embed = pos_embed.view(1, embed_size, max_length)
        self.register_buffer("pos_embed", pos_embed)
        
    def forward(self, x):
        # x is of shape [Batch, Channels, Time]
        x = x + self.pos_embed[:x.size(0)]
        return self.dropout(x)

class ConvUpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvUpBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)
    
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout, activation = 0.2):
        super(ConvBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            torch.nn.LeakyReLU(activation, inplace = True),
            torch.nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.block(x)

class ConvClassifierModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvClassifierModule, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True),
        )
    
    def forward(self, x):
        return self.block(x)
