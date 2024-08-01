import torch
import matplotlib.pyplot as plt
import os
from time import sleep
from IPython.display import clear_output
from tqdm import tqdm

class LearningRateScheduler(object):
    def __init__(self, base_lr, factor):
        self.base_lr = base_lr
        self.factor = factor
        
    def __call__(self, epoch):
        return self.base_lr / (1 + self.factor * epoch)
    
class LossTracker(object):
    def __init__(self):
        self.gen_loss = []
        self.critic_loss = []
        self.class_loss = []
        self.class_acc = []
        
    def append(self, gen_loss, critic_loss, class_loss, class_acc):
        self.gen_loss.append(gen_loss)
        self.critic_loss.append(critic_loss)
        self.class_loss.append(class_loss)
        self.class_acc.append(class_acc)
    
def generate_test_idx(span, slices, n, start=0):
    '''Generate n random indices from start to start + span * slices
    
    Used for generating random indices for test set'''
    sample = torch.randperm(span)[:n] + start
    for i in range(1, slices):
        sample = torch.cat([sample, torch.randperm(span)[:n] + start + i * span], dim=0)
    return sample

def gradient_penalty(batch_size, critic, real_img, fake_img, gp_weight, device = "cpu"):
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    diff = fake_img - real_img
    interpolates = real_img + (alpha * diff)
    interpolates = interpolates.to(device)
    
    pred = critic(interpolates)
    grads = torch.autograd.grad(outputs = pred, inputs = interpolates, grad_outputs = torch.ones_like(pred), create_graph = True, retain_graph = True, only_inputs = True)[0]
    norm = torch.sqrt(torch.sum(torch.square(grads), dim = [1, 2, 3]))
    gp = torch.mean(torch.square(norm - 1.0))
    return gp * gp_weight

def standardize_fn(data, mean, std):
    # [Conditions, EEGChannels, Time, 1]
    return ((data - mean) / std).float()

def normalize_to_neg_1_pos_1(img):
    return (img / 127.5) - 1

def create_char74k_tensor_set(data, test_data = False, save_dir : str = None, test_idx = None):
    if test_idx is None and test_data:
        test_idx = generate_test_idx(1016, 10, 203)
        if save_dir:
            if os.path.exists(save_dir):
                raise FileExistsError(f"{save_dir} already exists. Did you mean to overwrite it?")
            torch.save(test_idx, save_dir)
    elif test_data:
        mask = torch.ones(len(data), dtype=bool)
        mask[test_idx] = False
        training_data_subset = torch.utils.data.Subset(data, torch.nonzero(mask).squeeze())
        test_data_subset = torch.utils.data.Subset(data, torch.nonzero(~mask).squeeze())
        
        training_data = torch.stack([data[i] for i in training_data_subset.indices])
        test_data = torch.stack([data[i] for i in test_data_subset.indices])
        return training_data, test_data
    training_data = torch.stack([data[i] for i in range(len(data))])
    return training_data

def get_rand_imgs_from_label(imgs, label, n_span, device = "cpu"):
    '''Get batch_size random images that are of label'''
    batch_size = label.shape[0]
    class_indices = torch.argmax(label, dim=1)
    
    start_indices = class_indices * n_span
    random_offsets = torch.randint(0, n_span, (batch_size,)).to(device)
    random_indices = start_indices + random_offsets
    random_indices = random_indices.to(imgs.device)
    
    sampled_imgs = imgs[random_indices]
    return sampled_imgs

def train_EEGGAN(train_dataloader, imgs, gen, critic, classifier, classifier_loss_fn, optimizer_gen, optimizer_critic, epoch, gp_weight, class_weight = 0.9, critic_loops = 3, n_span = 813, device = "cpu", test_data = None, track_losses = None):
    for param in classifier.parameters():
        param.requires_grad = False
    classifier.eval()
    gen.to(device)
    critic.to(device)
    classifier.to(device)
    
    gen_loss = 0
    critic_loss = 0
    class_loss = 0
    class_accuracy = 0
    
    train_progress = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}")

    for (x, y) in train_progress:
        for p in critic.parameters():
            p.requires_grad = True
        x = x.to(device)
        y = y.to(device)
        
        for _ in range(critic_loops):
            real_imgs = get_rand_imgs_from_label(imgs, y, n_span, device).to(device)
            optimizer_critic.zero_grad()
            fake_imgs = gen(x)

            pred = classifier(real_imgs)
            real_class_loss = classifier_loss_fn(pred, y)
            pred = classifier(fake_imgs)
            fake_class_loss = classifier_loss_fn(pred, y)
            
            critic_loss = torch.mean(critic(fake_imgs)) - torch.mean(critic(real_imgs))
            critic_gp = gradient_penalty(x.shape[0], critic, real_imgs, fake_imgs, gp_weight, device = device)
            critic_loss += critic_gp + (real_class_loss + fake_class_loss) * class_weight
            critic_loss.backward()
            optimizer_critic.step()
        
        for p in critic.parameters():
            p .requires_grad = False
            
        optimizer_gen.zero_grad()
        fake_imgs = gen(x)
        pred = classifier(fake_imgs)
        class_loss = classifier_loss_fn(pred, y)
        gen_loss = -torch.mean(critic(fake_imgs)) + class_loss * class_weight
        gen_loss.backward()
        optimizer_gen.step()
        
        train_progress.set_postfix({
            'Generator Loss': f"{gen_loss.item():.4f}",
            'Critic Loss': f"{critic_loss.item():.4f}",
            'Classifier Loss': f"{class_loss.item() * class_weight:.4f}"
        })
        gen_loss += gen_loss.item()
        critic_loss += critic_loss.item()
        class_loss += class_loss.item()
        class_accuracy += torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)).item()
        
    gen_loss /= len(train_dataloader)
    critic_loss /= len(train_dataloader)
    class_loss /= len(train_dataloader)
    class_accuracy /= (len(train_dataloader) * train_dataloader.batch_size)
    
    train_progress.set_postfix({
        'Generator Loss': f"{gen_loss:.4f}",
        'Critic Loss': f"{critic_loss:.4f}",
        'Classifier Loss': f"{class_loss:.4f}",
        'Generator Class Accuracy': f"{class_accuracy:.4f}%"
    })
    
    if track_losses:
        track_losses.append(gen_loss, critic_loss, class_loss, class_accuracy)
        
    if test_data:
        eeg, labels = next(iter(test_data))
    else:
        eeg, labels = next(iter(train_dataloader))
    label_mapping = {0: 'A', 1: 'C', 2: 'F', 3: 'H', 4: 'J', 5: 'M', 6: 'P', 7: 'S', 8: 'T', 9: 'Y'}
    
    eeg = eeg.to(device)
    labels = labels.to(device)
    imgs = gen(eeg)
    n_cols = 8
    n_rows = 4
    plt.figure(figsize=(5, 5))
    
    for index, image in enumerate(imgs[:n_cols*n_rows]):
        plt.subplot(n_rows, n_cols, index+1)
        plt.imshow((image.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2, cmap="gray")
        plt.axis("off")
        
        # Get the label index and corresponding label text
        label_index = torch.argmax(labels[index]).item()
        label_text = label_mapping.get(label_index, 'Unknown')
        
        # Set the title of the subplot to the label text
        plt.title(label_text)
    
    plt.show()
    
def train_classifier(train_dataloader, model, optimizer, loss_fn, epoch, eeg_classifier=False, test=False, test_dataloader=None, device='cpu', save=False, scheduler=None, save_dir="weights/vgg16", num_files=3):
    if test and not test_dataloader:
        raise ValueError("test_dataloader is required when test is True")
    
    num_correct = 0
    total = 0
    train_progress = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}")

    for (x, y) in train_progress:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        if eeg_classifier:
            pred = model(x, classification=True)
        else:
            pred = model(x)
            
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        total += y.shape[0]
        num_correct += torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)).item()
        
        
        train_progress.set_postfix({
            'Loss': f"{loss.item():.6f}",
            'Accuracy': f"{num_correct / total * 100:.4f}%"
        })
    
    if scheduler:
        scheduler.step()
    
    if test:
        test_progress = tqdm(total=1, desc="Testing")
        test_loss, test_accuracy = test_classifier(test_dataloader, model, loss_fn, eeg_classifier, device, save, save_dir, num_files)
        test_progress.set_postfix({
            'Test Loss': test_loss,
            'Test Accuracy': f"{test_accuracy * 100:.4f}%"
        })
        test_progress.update(1)
        sleep(1)
    
def test_classifier(test_dataloader, model, loss_fn, eeg_classifier = True, device = "cpu", save = False, save_dir = "weights/vgg16", num_files : int = 3):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            if eeg_classifier:
                pred = model(x, classification = True)
            else:
                pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            total_correct += torch.sum(torch.argmax(pred, dim = 1) == torch.argmax(y, dim = 1)).item()
            total += y.shape[0]
    model.train()
    accuracy = total_correct / total
    if save:
        # Ensure the save directory exists
        save_model(model, save_dir, accuracy, num_files)
    
    return total_loss / len(test_dataloader), accuracy

def save_model(model, save_dir, accuracy, num_files = 3):
    os.makedirs(save_dir, exist_ok=True)
        
    # Get the list of saved files
    files = os.listdir(save_dir)
    
    if len(files) < num_files:
        # Save the model if there are less than num_files saved
        torch.save(model.state_dict(), f"{save_dir}/{accuracy:.4f}.pt")
    else:
        # Extract accuracies from filenames
        accuracies = [float(f.split('.pt')[0]) for f in files]
        min_accuracy = min(accuracies)
        
        if accuracy > min_accuracy:
            # Find the file with the lowest accuracy and replace it
            min_index = accuracies.index(min_accuracy)
            os.remove(os.path.join(save_dir, files[min_index]))
            torch.save(model.state_dict(), f"{save_dir}/{accuracy:.5f}.pt")
            
def train_GAN(train_dataloader, gen, critic, optimizer_gen, optimizer_critic, epoch, gp_weight, device = "cpu"):
    for batch, (x, y) in enumerate(train_dataloader):
        for p in critic.parameters():
            p.requires_grad = True
        
        eeg_signals = x.to(device)
        real_imgs = y.to(device)
        for _ in range(3):
            optimizer_critic.zero_grad()
            fake_imgs = gen(eeg_signals)
            critic_loss = torch.mean(critic(fake_imgs)) - torch.mean(critic(real_imgs))
            critic_gp = gradient_penalty(x.shape[0], critic, real_imgs, fake_imgs, gp_weight, device = device)
            critic_loss += critic_gp
            critic_loss.backward()
            optimizer_critic.step()
        
        for p in critic.parameters():
            p .requires_grad = False
        optimizer_gen.zero_grad()
        fake_imgs = gen(eeg_signals)
        gen_loss = -torch.mean(critic(fake_imgs))
        gen_loss.backward()
        optimizer_gen.step()
            
        if batch % 20 == 0:
            clear_output(wait=True)
            print("Epoch: {} Batch: {}\nCritic Loss: {} Gen Loss: {}".format(epoch, batch, critic_loss.item(), gen_loss.item()))
            n_cols = 8
            n_rows = 8
            plt.figure(figsize=(10, 10))
            for index, image in enumerate(fake_imgs[:n_cols*n_rows]):
                plt.subplot(n_rows, n_cols, index+1)
                plt.imshow((image.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2, cmap="gray")
                plt.axis("off")
            plt.show()
            
def create_interpolations(z1, z2, increment = 0.1):
    alphas = torch.arange(0, 1.1, increment)
    interpolated_tensors = []
    
    # Perform the interpolation for each alpha
    for alpha in alphas:
        z = alpha * z1 + (1 - alpha) * z2
        interpolated_tensors.append(z)
        
    interpolated_tensor = torch.stack(interpolated_tensors).squeeze()
    return interpolated_tensor