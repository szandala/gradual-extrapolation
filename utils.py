import torch
from torch.nn.functional import interpolate

import numpy as np
import torch.nn as nn
import os
import glob
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224))
])

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def read_images_2_batch():
    image_files = glob.glob("./samples/*.jpg")
    image_files.sort()

    input_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                    for f in image_files]
    input_batch = torch.stack([normalize(crop(image))
                               for image in input_images])

    return input_batch


def get_category_IDs(model, input_batch):
    percentage = nn.Softmax(dim=1)
    output = percentage(model(input_batch))

    listed_all_outputs = [[(i, val)
                           for i, val in enumerate(o.tolist())] for o in output]
    best_fits = [sorted(sing_out, key=lambda o: o[1])[-1]
                 for sing_out in listed_all_outputs]
    return [o[0] for o in best_fits]



def _imsc(img, *args,  interpolation="lanczos", **kwargs):
    r"""Rescale and displays an image represented as a img.

    The function scales the img :attr:`im` to the [0 ,1] range.
    The img is assumed to have shape :math:`3\times H\times W` (RGB)
    :math:`1\times H\times W` (grayscale).

    Args:
        img (:class:`torch.Tensor` or :class:`PIL.Image`): image.
        quiet (bool, optional): if False, do not display image.
            Default: ``False``.
        lim (list, optional): maximum and minimum intensity value for
            rescaling. Default: ``None``.
        interpolation (str, optional): The interpolation mode to use with
            :func:`matplotlib.pyplot.imshow` (e.g. ``"lanczos"`` or
            ``"nearest"``). Default: ``"lanczos"``.

    Returns:
        Nothing, changes in-place
    """

    with torch.no_grad():
        lim = [img.min(), img.max()]
        img = img - lim[0]  # also makes a copy
        img.mul_(1 / (lim[1] - lim[0]))
        img = torch.clamp(img, min=0, max=1)
        # print(img.shape)
        bitmap = img.expand(*img.shape).permute(1, 2, 0).cpu().numpy()
        handle = plt.imshow(
            bitmap, *args, interpolation=interpolation, **kwargs)
        curr_ax = plt.gca()
        curr_ax.axis("off")


def plot_example(input,
                 saliency,
                 method,
                 category_id,
                 save_path=None):
    """Plot an example.

    Args:
        input (:class:`torch.Tensor`): 4D tensor containing input images.
        saliency (:class:`torch.Tensor`): 4D tensor containing saliency maps.
        method (str): name of saliency method.
        category_id (int): ID of ImageNet category.
        show_plot (bool, optional): If True, show plot. Default: ``False``.
        save_path (str, optional): Path to save figure to. Default: ``None``.
    """
    from torchray.benchmark.datasets import IMAGENET_CLASSES

    if isinstance(category_id, int):
        category_id = [category_id]

    batch_size = len(input)

    plt.clf()
    for i in range(batch_size):
        class_i = category_id[i % len(category_id)]
        plt.subplot(batch_size, 2, 1 + 2 * i)
        plt.tight_layout(pad=0.5)
        _imsc(input[i])

        plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative co-ordinates.
        plt.rcParams['axes.titlepad'] = 1  # pad is in points...

        plt.title("input image", fontsize=6)
        plt.subplot(batch_size, 2, 2 + 2 * i)
        _imsc(saliency[i], interpolation="bilinear", cmap="jet")

        cls_name = IMAGENET_CLASSES[class_i].split(",")[0]
        plt.title("{} for {}".format(method, cls_name), fontsize=6)

    # Save figure if path is specified.
    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        # Create directory if necessary.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ext = os.path.splitext(save_path)[1].strip(".")
        plt.savefig(save_path, format=ext, dpi=300, bbox_inches="tight")
