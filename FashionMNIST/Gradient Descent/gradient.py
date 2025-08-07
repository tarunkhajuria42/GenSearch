import torch
from fmnist_generator import *

import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from position_utils import *
from torchvision import transforms
g_obj = fmnist_generator()

# ----------- MOCK GENERATOR FUNCTION (replace with your trained GAN) -----------
def generator(z):
    # This is a placeholder for the GAN generator. Replace with actual generator.
    image = g_obj.generate_samples(z)
    return image

# ----------- DOT HEATMAP (Differentiable Target) -----------
def create_dot_heatmap(dot_coords, size=(160, 160), sigma=2.0, device='cpu'):
    heatmap = torch.zeros(size, device=device)
    xs = torch.arange(0, size[1], device=device).view(1, -1)
    ys = torch.arange(0, size[0], device=device).view(-1, 1)
    for (x, y) in dot_coords:
        heatmap += torch.exp(-((xs - x)**2 + (ys - y)**2) / (2 * sigma**2))
    return heatmap.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

# ----------- SOBEL EDGE DETECTION (Differentiable) -----------
def sobel_edges(image,device='cpu'):
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device = device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32,device = device).view(1, 1, 3, 3)
    edge_x = F.conv2d(image, sobel_x, padding=1)
    edge_y = F.conv2d(image, sobel_y, padding=1)
    edge_mag = torch.sqrt(torch.clamp(edge_x**2 + edge_y**2, min=1e-6))
    return edge_mag

# ----------- MAIN OPTIMIZATION -----------
def optimize_latent(dot_coords,dots, steps=100, lr=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = g_obj.generate_latents(1)
    z.requires_grad = True
    optimizer = torch.optim.Adam([z], lr=lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.1,        # minimum learning rate
        max_lr=0.5,         # maximum learning rate
        step_size_up=1000,   # number of training steps to reach max_lr
        mode='triangular',   # can also be 'triangular2' or 'exp_range'
        cycle_momentum=False # set to False if not using momentum-based optimizer
    )

    target_heatmap = create_dot_heatmap(dot_coords, device=device)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        gen_image = generator(z)
        resize = transforms.Resize((160, 160))  # e.g., (256, 256)
        gen_image = resize(gen_image)
        edge_map = sobel_edges(gen_image)
        mask = target_heatmap!= 0
        masked_edge = edge_map*mask
        
        #loss = F.mse_loss(, target_heatmap)   # maximize alignment
        loss = -torch.sum(masked_edge * target_heatmap)
        loss.shape
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}",scheduler.get_last_lr(),)
            plt.figure()
            plt.imshow(edge_map.squeeze().detach().cpu().numpy(), cmap='gray')
            edge = edge_map.squeeze().detach().cpu().numpy()
            normalised = edge*255/edge.max()
            count = len(points_on_image(normalised,dots,2)[0])
            print(count)
            

    return z.detach(), gen_image.detach(), target_heatmap.detach(), sobel_edges(gen_image).detach(), losses

# ----------- RUN AND VISUALIZE -----------
if __name__ == '__main__':
    inputs = np.load('../dataset/fashion_constellation_test_sel_11.npz') # point to your input constellation test set
    correct = 0
    #classifier = mnist_classifier()
    for i, image in enumerate(inputs['images']): 
        label = inputs['labels'][i]
        base_img = image.astype(np.uint8)
        dots = stimuli_dots(base_img)
        X,Y = find_dot_centres(dots)
        dot_coords = list(zip(X,Y))
        z_opt, gen_image, target_map, edge_map, losses = optimize_latent(dot_coords,dots,steps = 5000, lr = 0.5)
        #predicted = classifier.predict( cv2.resize(gen_image.squeeze().cpu().numpy(), (28,28)))
        edge = edge_map.squeeze().cpu().numpy()
        normalised = edge*255/edge.max()
        count = len(points_on_image(normalised,dots,3)[0])
        print(count)
    