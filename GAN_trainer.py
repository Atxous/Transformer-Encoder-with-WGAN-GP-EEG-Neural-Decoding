import torch
import torchvision
from torch.utils.data import DataLoader
from data.data import char74kDataset, EEGClassificationDataset
from model.model import load_vgg16, EEGGenerator, EEGCritic, EEGEncoder
from helper_fn import create_char74k_tensor_set, normalize_to_neg_1_pos_1, train_EEGGAN, LossTracker

NUM_CLASSES = 10
BATCH_SIZE = 128
GP_WEIGHT = 8
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = torch.load("statistics/mean.pt")
std = torch.load("statistics/std.pt")
eeg_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])
img_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64), antialias=True),
    normalize_to_neg_1_pos_1
])

data = EEGClassificationDataset("EEG_img_data\data\eeg\char\data.pkl", train = True, transforms = eeg_transforms)
imgs = char74kDataset("EEG_img_data\char74k\characters", imgs_only = True, transforms = img_transforms)
test_idx = torch.load("statistics/char74k_test_idx.pt")
imgs_train= create_char74k_tensor_set(imgs, save_dir = "statistics/char74k_test_idx.pt")
eeg_dataloader = DataLoader(data, batch_size = BATCH_SIZE, shuffle = True)
eeg_test_dataloader = DataLoader(data, batch_size = 32, shuffle = False)

encoder = EEGEncoder(d_model = 100, timesteps = 32, nhead = 7, eeg_channels = 14, dim_feedforward = 8, dropout = 0.6)
encoder.load_state_dict(torch.load("weights/EEGEncoder/best_weights.pt"))
gen = EEGGenerator(d_model = 100, max_channels = 512, out_channels = 1, kernel_size = 4, stride = 2, padding = 1, num_blocks = 3, encoder = encoder)
critic = EEGCritic(1, 128, 4, 2, 1, 3, 0.3)
classifier = load_vgg16(NUM_CLASSES, 1, weights = r"weights\vgg16\best_weights_vgg16.pt")

classifier_loss_fn = torch.nn.CrossEntropyLoss()
gen_optim = torch.optim.RMSprop(gen.parameters(), lr = 0.0002)
critic_optim = torch.optim.RMSprop(critic.parameters(), lr = 0.0002)

loss_tracker = LossTracker()
for epoch in range(EPOCHS):
    train_EEGGAN(eeg_dataloader, imgs_train, gen, critic, classifier, classifier_loss_fn, gen_optim, critic_optim, epoch, GP_WEIGHT, class_weight = 1, device = device, n_span = 1016, track_losses = loss_tracker)
