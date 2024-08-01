import torch
import torchvision
from .blocks import EEGPositionalEncoder, ConvBlock, ConvUpBlock, ConvClassifierModule, GaussianLayer

class EEGEncoder(torch.nn.Module):
    def __init__(self, d_model, timesteps, nhead, eeg_channels, dim_feedforward, dropout, num_classes = 10):
        super(EEGEncoder, self).__init__()
        # [Batch, Channels, Timesteps]
        self.pos_encoding = EEGPositionalEncoder(timesteps, eeg_channels)
        self.multihead = torch.nn.MultiheadAttention(eeg_channels, nhead, dropout=dropout, batch_first=True)
        self.layer_norm1 = torch.nn.LayerNorm(eeg_channels)
        self.conv1 = torch.nn.Sequential(
            ConvClassifierModule(1, timesteps, (14, 1), 1, 0),
            ConvClassifierModule(timesteps, d_model, (1, timesteps), 1, 0),
            torch.nn.Flatten()
        )
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_feedforward, d_model),
            torch.nn.Dropout(dropout)
        )
        self.classification = torch.nn.Linear(d_model, num_classes)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.d_model = d_model
    
    def forward(self, x, classification = False):
        x = x.view(-1, x.shape[-1], x.shape[-2])
        x = self.pos_encoding(x)
        skip = x
        x, _ = self.multihead(x, x, x)
        x = torch.add(x, skip)
        x = self.layer_norm1(x)
        
        # Add channel dimension
        x = x.view(-1, 1, x.shape[-1], x.shape[-2])
        x = self.conv1(x)
        
        skip = x
        x = self.ff(x)
        x = torch.add(x, skip)
        x = self.layer_norm2(x)
        
        if classification:
            return self.classification(x)
        
        return x
    
class EEGGenerator(torch.nn.Module):
    def __init__(self, d_model, max_channels, out_channels, kernel_size, stride, padding, num_blocks, encoder):
        super(EEGGenerator, self).__init__()
        self.encoder = encoder
        # make the encoder untrainable and in evaluation mode
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.gaussian = GaussianLayer(d_model)
        self.input = ConvUpBlock(d_model, max_channels, kernel_size, 1, 0)
        self.generator = torch.nn.ModuleList([ConvUpBlock(max_channels // (2 ** i), max_channels // (2 ** (i + 1)), kernel_size, stride, padding) for i in range(num_blocks)])
        self.output = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(max_channels // (2 ** num_blocks), out_channels, kernel_size, stride, padding),
            torch.nn.Tanh()
        )
        
    def forward(self, x):
        # x is of shape [Batch, Channels, Time]
        x = self.encoder(x)
        # x is of shape [Batch, Channels]
        noise = torch.randn_like(x)
        x = self.gaussian(x, noise)
        # Add spatial dimensions
        x = x.view(-1, x.shape[-1], 1, 1)
        x = self.input(x)
        
        for block in self.generator:
            x = block(x)
            
        return self.output(x)
    
    def generate_img_given_eeg(self, x, noise):
        x = self.gaussian(x, noise)
        x = x.view(-1, x.shape[-1], 1, 1)
        x = self.input(x)
        
        for block in self.generator:
            x = block(x)
            
        return self.output(x)
    
class EEGCritic(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_blocks, dropout):
        super(EEGCritic, self).__init__()
        self.input = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, dropout)
        self.critic = torch.nn.ModuleList([ConvBlock(out_channels * (2 ** i), out_channels * (2 ** (i + 1)), kernel_size, stride, padding, dropout) for i in range(num_blocks)])
        self.output = torch.nn.Conv2d(out_channels * (2 ** num_blocks), 1, kernel_size, 1, 0)
        
    def forward(self, x):
        x = self.input(x)
        
        for block in self.critic:
            x = block(x)
            
        return self.output(x)
    
def load_vgg16(n_classes, in_channels, weights : str = None):
    vgg16 = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
    # replace the initial layer with number of input channels 
    vgg16.features[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=1, padding=1)
    # replace the classifier with a new one
    vgg16.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(4096, n_classes)
    )
    if weights:
        vgg16.load_state_dict(torch.load(weights))
        
    return vgg16
    