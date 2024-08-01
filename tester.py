# %%
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model.model import EEGEncoder, EEGGenerator
from data.data import EEGClassificationDataset
from data.transformations import Standardize
from helper_fn import create_interpolations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = torch.load("statistics/mean.pt")
std = torch.load("statistics/std.pt")
transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std)
])

encoder = EEGEncoder(d_model = 100, timesteps = 32, nhead = 7, eeg_channels = 14, dim_feedforward = 8, dropout = 0.6)
encoder.to(device)

# load encoder weights
encoder.load_state_dict(torch.load(r"weights\EEGEncoder\best_weights.pt"))

gen = EEGGenerator(d_model = 100, max_channels = 512, out_channels = 1, kernel_size = 4, stride = 2, padding = 1, num_blocks = 3, encoder = encoder)
gen.to(device)
gen.load_state_dict(torch.load(r"weights\EEGGAN\gen.pt"))

encoder.eval()
gen.eval()

data = EEGClassificationDataset("EEG_img_data\data\eeg\char\data.pkl", train = True, transforms = transforms)
noise = torch.randn(1, 100).to(device)
# %%
interpolated_tensors = []
with torch.no_grad():
    (x1, y1), (x2, y2) = data[523], data[214]
    z1, z2 = encoder(x1.unsqueeze(0).to(device)), encoder(x2.unsqueeze(0).to(device))
    interpolated_tensor = create_interpolations(z1, z2, 0.15)
    interpolated_tensors.append(interpolated_tensor)
    
    (x1, y1), (x2, y2) = data[54], data[65]
    z1, z2 = encoder(x1.unsqueeze(0).to(device)), encoder(x2.unsqueeze(0).to(device))
    interpolated_tensor = create_interpolations(z1, z2, 0.15)
    interpolated_tensors.append(interpolated_tensor)
    (x1, y1), (x2, y2) = data[12], data[75]
    z1, z2 = encoder(x1.unsqueeze(0).to(device)), encoder(x2.unsqueeze(0).to(device))
    interpolated_tensor = create_interpolations(z1, z2, 0.15)
    interpolated_tensors.append(interpolated_tensor)
    (x1, y1), (x2, y2) = data[53], data[823]
    z1, z2 = encoder(x1.unsqueeze(0).to(device)), encoder(x2.unsqueeze(0).to(device))
    interpolated_tensor = create_interpolations(z1, z2, 0.15)
    interpolated_tensors.append(interpolated_tensor)
    (x1, y1), (x2, y2) = data[671], data[16]
    z1, z2 = encoder(x1.unsqueeze(0).to(device)), encoder(x2.unsqueeze(0).to(device))
    interpolated_tensor = create_interpolations(z1, z2, 0.15)
    interpolated_tensors.append(interpolated_tensor)

interpolated_tensor = torch.stack(interpolated_tensors, dim=0)
interpolated_tensor = interpolated_tensor.view(-1, 100)
# %%
output = gen.generate_img_given_eeg(interpolated_tensor, noise)
# %%
import matplotlib.pyplot as plt

# Convert the output to a numpy array and squeeze the channel dimension
images = output.cpu().detach().numpy().squeeze()

# Create a figure with 11 subplots in a single row
fig, axes = plt.subplots(5, 8, figsize=(15, 8))

# Plot each image in the corresponding subplot
for i in range(5):
    for j in range(8):
        ax = axes[i, j]
        ax.imshow(images[i * 8 + j], cmap='gray')
        ax.axis('off')  # Remove axes

# Adjust the layout to remove any remaining whitespace
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

# %%
# %%

# %%
