import numpy as np
import io
import os
import torchvision.transforms as T
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl

from PIL import Image

from .helper import get_toy_target_dictionary

def show_max_activation(image, segmentations, class_id):
    nat_image = get_unnormalized_image(image)

    mask = segmentations[0][class_id].numpy()
    max_pixel_coords = np.unravel_index(mask.argmax(), mask.shape)

    circle = plt.Circle(max_pixel_coords[::-1], 10, fill=False, color='red')
    fig, ax = plt.subplots(1)
    ax.imshow(np.stack(nat_image.squeeze(), axis=2))
    ax.add_patch(circle)

    plt.show()

def save_mask(mask, filename, dataset):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    path_file = os.path.splitext(filename)[0]

    if dataset == 'COLOR':
        t = torch.zeros((200,200), device=mask.device)
        b, h, w = mask.size()
        s = 200 // h
        for i in range(h):
            for j in range(w):
                t[s*i:i*s+s, s*j:j*s+s] = torch.ones((s, s), device=mask.device) * mask[0, i,j]
        mask = t

    img = mask.detach().cpu().numpy().squeeze()

    plt.imsave(path_file + ".png", img, cmap='gray', vmin=0, vmax=1, format="png")
    np.savez_compressed(path_file + ".npz", img)

def save_masked_image(image, mask, filename, dataset):
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    path_file = os.path.splitext(filename)[0]

    if dataset != 'COLOR':
        image = get_unnormalized_image(image)
    masked_nat_im = get_masked_image(image, mask)


    

    if dataset != 'COLOR':
        plt.imsave(path_file + ".png", np.stack(masked_nat_im.detach().cpu().squeeze(), axis=2), format="png")
    else:
        b,c, h, w = masked_nat_im.size()
                

        if c == 1:
            t = torch.zeros((200,200), device=masked_nat_im.device)
            dims = (20, 20)
            kwargs = {
                'cmap':'gray',
                'vmin': 0,
                'vmax': 1
            }
        else:
            s = 200 // h
            t = torch.zeros((200,200, 3), device=masked_nat_im.device)
            dims = (s, s, 3)
            kwargs = {}

        for i in range(h):
            for j in range(w):
                t[s*i:i*s+s, s*j:j*s+s] = torch.ones(dims, device=masked_nat_im.device) * masked_nat_im[0,0 if c == 1 else 0:3, i,j]

        plt.imsave(path_file + ".png", t.detach().cpu().squeeze().numpy() / 255., format="png", **kwargs)

def save_image(image, filename, dataset):
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    path_file = os.path.splitext(filename)[0]

    if dataset != 'COLOR':
        image = get_unnormalized_image(image)
    

    if dataset != 'COLOR':
        plt.imsave(path_file + ".png", np.stack(image.cpu().squeeze(), axis=2), format="png")
    else:
        b,c, h, w = image.size()

        if c == 1:
            t = torch.zeros((200,200), device=image.device)
            dims = (20, 20)
            kwargs = {
                'cmap':'gray',
                'vmin': 0,
                'vmax': 1
            }
        else:
            s = 200 // h
            t = torch.zeros((200,200, 3), device=image.device)
            dims = (s, s, 3)
            kwargs = {}

        for i in range(h):
            for j in range(w):
                t[s*i:i*s+s, s*j:j*s+s] = torch.ones(dims, device=image.device) * image[0,0 if c == 1 else 0:3, i,j]

        plt.imsave(path_file + ".png", t.detach().cpu().squeeze().numpy() / 255., format="png", **kwargs)

def show_image_and_masked_image(image, mask):
    nat_image = get_unnormalized_image(image)
    masked_nat_im = get_masked_image(nat_image, mask)

    fig = get_fullscreen_figure_canvas("Image and masked image")
    fig.add_subplot(1, 2, 1)
    show_image(nat_image)

    fig.add_subplot(1, 2, 2)
    show_image(masked_nat_im)

    plt.show()


def save_all_class_masks(segmentations, filename, dataset):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    filename = os.path.splitext(filename)[0]

    all_class_masks = segmentations.transpose(0, 1).sigmoid()

    num_classes = len(get_target_labels(dataset, include_background_class=False))
    mpl.rcParams["figure.figsize"] = (40,10*(num_classes // 5 + 1))
    fig = plt.figure()
    fig.suptitle('All class masks')
    for i in range(all_class_masks.size()[0]): #loop over all classes
        add_subplot_with_class_mask(fig, i, dataset)
        plt.imshow((all_class_masks[i].detach().cpu().numpy().squeeze()), cmap='gray', vmin=0, vmax=1)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    im = Image.open(img_buf)
    im.save(filename + '.png', format='png')

    #img = mask.detach().cpu().numpy().squeeze()

    #plt.imsave(path_file + ".png", img, cmap='gray', vmin=0, vmax=1, format="png")
    #np.savez_compressed(path_file + ".npz", img)

    img_buf.close()

def save_all_class_masked_images(image, segmentations, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    filename = os.path.splitext(filename)[0]

    nat_image = get_unnormalized_image(image)
    all_class_masks = segmentations.transpose(0, 1).sigmoid()

    fig = get_fullscreen_figure_canvas("All class masks")
    for i in range(all_class_masks.size()[0]): #loop over all classes
        masked_nat_im = get_masked_image(nat_image, all_class_masks[i]).detach().cpu()
        add_subplot_with_class_mask(fig, i)
        show_image(masked_nat_im)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    im = Image.open(img_buf)
    im.save(filename, format='png')

    img_buf.close()

def show_target_class_masks(image, segmentations, targets):
    nat_image = get_unnormalized_image(image)
    all_class_masks = segmentations.transpose(0, 1).sigmoid()

    fig = get_fullscreen_figure_canvas("Target class masks")
    for i in range(all_class_masks.size()[0]): #loop over all classes
        if targets[0][i] == 1.0:
            masked_nat_im = get_masked_image(nat_image, all_class_masks[i])
            add_subplot_with_class_mask(fig, i)
            show_image(masked_nat_im)

def show_most_likely_class_masks(image, segmentations, logits, threshold=0.0):
    nat_image = get_unnormalized_image(image)
    all_class_masks = segmentations.transpose(0, 1).sigmoid()

    fig = get_fullscreen_figure_canvas("Predicted class masks")
    for i in range(all_class_masks.size()[0]): #loop over all classes
        if logits[0][i] >= threshold:
            masked_nat_im = get_masked_image(nat_image, all_class_masks[i])
            add_subplot_with_class_mask(fig, i)
            show_image(masked_nat_im)

def get_unnormalized_image(image):
    inverse_transform = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                   T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])

    nat_image = inverse_transform(image)

    return nat_image

def get_masked_image(image, mask):
    masked_image = mask.unsqueeze(1) * image

    return masked_image

def get_fullscreen_figure_canvas(title):
    mpl.rcParams["figure.figsize"] = (40,40)
    fig = plt.figure()
    fig.suptitle(title)

    return fig

def add_subplot_with_class_mask(fig, class_id, dataset):
    target_labels = get_target_labels(dataset, include_background_class=False)
    rows = len(target_labels) // 5 + 1

    axis = fig.add_subplot(rows, 5, class_id+1)
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.title.set_text(target_labels[class_id])

def show_image(image):
    plt.imshow(np.stack(image.squeeze(), axis=2))

def get_target_labels(dataset, include_background_class):
    if dataset == 'VOC':
        if include_background_class:
            targets = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
                    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        else:
            targets = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
                    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset == 'TOY':
        targets = list(get_toy_target_dictionary(include_background_class, 'texture').keys())

    return targets


def save_background_logits(logits, path_file):
    plt.figure()
    x = np.arange(len(logits))
    plt.plot(x, logits)
    plt.title('Background pass logits')
    plt.savefig(path_file)
