import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model.model import EEGEncoder
from data.data import EEGClassificationDataset
from helper_fn import train_classifier

EPOCHS = 20

def lr_lambda(epoch):
    # LR to be 0.001 * (1/1+0.04*epoch)
    base_lr = 0.025
    factor = 0.02
    return base_lr/(1+factor*(epoch + 100))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the mean and std for standardization
mean = torch.load("statistics/mean.pt")
std = torch.load("statistics/std.pt")
transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

data = EEGClassificationDataset("EEG_img_data\data\eeg\char\data.pkl", train = True, transforms = transforms)

test_data = EEGClassificationDataset("EEG_img_data\data\eeg\char\data.pkl", train = False, transforms = transforms)
train_dataloader = DataLoader(data, batch_size = 32, shuffle = False)
test_dataloader = DataLoader(test_data, batch_size = 32, shuffle = False)

classifier = EEGEncoder(d_model = 100, timesteps = 32, nhead = 7, eeg_channels = 14, dim_feedforward = 8, dropout = 0.62)
classifier.to(device)
classifier.load_state_dict(torch.load("weights/best_weights.pt"))

optimizer = torch.optim.AdamW(classifier.parameters(), lr = 0.025, weight_decay= 0.085)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
loss_fn = torch.nn.CrossEntropyLoss()

for i in range(EPOCHS):
    train_classifier(train_dataloader, classifier, optimizer, loss_fn, i, test = True, test_dataloader = test_dataloader, device = device, save = True)

