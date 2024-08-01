# load VGG16 model
import torch
import torchvision
from data.data import char74kDataset
from helper_fn import generate_test_idx, normalize_to_neg_1_pos_1, train_classifier

NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg16 = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
# replace the classifier with a new one
vgg16.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(4096, NUM_CLASSES)
)
vgg16.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1)
vgg16.load_state_dict(torch.load("weights/vgg16/0.99015.pt"))
vgg16.to(device)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64), antialias=True),
    normalize_to_neg_1_pos_1
])

data = char74kDataset("EEG_img_data\char74k\characters", transforms = transforms)
test_idx = generate_test_idx(1016, 10, 203)
# Create a mask to exclude test_idx
mask = torch.ones(len(data), dtype=bool)
mask[test_idx] = False

# Filter data to exclude test_idx
training_data = torch.utils.data.Subset(data, torch.nonzero(mask).squeeze())
test_data = torch.utils.data.Subset(data, torch.nonzero(~mask).squeeze())
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size = 128, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 128, shuffle = False)

optim = torch.optim.NAdam(vgg16.parameters(), lr = 5e-4, weight_decay = 1e-3)
epochs = 30
loss_fn = torch.nn.CrossEntropyLoss()

# freeze the beginning of the model except the initial layer
frozen_layers = 0
for param in list(vgg16.features.parameters())[1:15]:
    param.requires_grad = False
    frozen_layers += 1

for i in range(epochs):
    train_classifier(train_dataloader, vgg16, optim, loss_fn, i, eeg_classifier = False, test = True, test_dataloader = test_dataloader, device = device, save = True, save_dir = "weights/vgg16")
    # if epoch is divisible by 6, unfreeze the topmost 4 layers
    if i % 6 == 0 and i != 0 and frozen_layers > 1:
        layers_to_unfreeze = 4
        for param in list(vgg16.features.parameters())[frozen_layers - 4:frozen_layers + 1]:
            param.requires_grad = True
        frozen_layers -= layers_to_unfreeze
    # if we've unfrozen 24 layers, there's only 1 left so we unfreeze everything
    elif frozen_layers == 1:
        for param in vgg16.features.parameters():
            param.requires_grad = True
        frozen_layers = 0

