from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.excitation_backprop import contrastive_excitation_backprop
from torchray.attribution.deconvnet import deconvnet

from utils import read_images_2_batch, get_category_IDs, plot_example, save_img
from gradual_extrapolation import GradualExtrapolator
from torchvision import models

import torch
from torch.nn.functional import interpolate
import sys
from torchsummary import summary

from statistics import median

def sum_pixels(saliency_map):
    # print(saliency_map.shape)
    saliency_map = interpolate(
                saliency_map,
                size=(input_batch.shape[2], input_batch.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

    # print(saliency_map.shape)
    # print(saliency_map.sum(dim=[3,2]))
    # print("===")
    print(saliency_map.mean(dim=[3,2,1,0]))
    # for img in saliency_map.tolist():

    #     s = 0
    #     for row in img[0]:

    #         for y in row:

    #             s+= y
    #     print(f"Sum is: {s}")
    return saliency_map

def count_nonzero(batch):
    imgs = []
    for img in batch:
        counter = 0
        for row in img[0]:
            for p in row:
                if p != 0:
                    counter+=1
        imgs.append(counter)
    return imgs

def trim_map(saliencies):
    trimmed_saliency = []
    for img in saliencies:
        median = img.flatten().quantile(dim=0, q=0.9) #[0]
        print(f"median for image is: {median}")
        trimmed_saliency.append(torch.where(img < median, torch.zeros(224, 224), img).clip(0,1))

    return torch.stack(trimmed_saliency)



if __name__ == "__main__":
    CLIP_INPUT = False
    TRIM_SALIENCES=True
    NET = "googlenet"
    layers = {
        "resnet50": ["layer4", "layer3.5"], # valuable
        "vgg11": ["features.18", "features.14"],
        "vgg16": ["features.28", "features.18"],# valuable
        "vgg19": ["features.34", "features.21"],
        "densenet161": ["features.norm5", "features.norm5"],
        "alexnet": ["features.8", "features.8"],
        "squeezenet1_0": ["features.11", "features.11"],
        "googlenet": ["inception5a", "inception3a"],# valuable
        "mobilenet_v2": ["features.16", "features.16"] #["features.16", "features.13"]
    }
    input_batch, filenames = read_images_2_batch()

    # TODO: loop over all common networks
    model = models.__dict__[NET](pretrained=True)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name)
    summary(model, (3, 224, 224))

    model.eval()
    category_id = get_category_IDs(model, input_batch, filenames)
    deconvnet(model, input_batch, category_id)

    print("Processing Grad-CAM...")
    saliency_grad_cam = grad_cam(
        model,
        input_batch,
        category_id,
        saliency_layer=layers[NET][0],
    )

    # saliency_grad_cam = (saliency_grad_cam - 1.0)*(-1.0)

    orig_saliency_grad_cam = saliency_grad_cam
    saliency_grad_cam = sum_pixels(saliency_grad_cam)
    if CLIP_INPUT:
        multiplier = torch.where(saliency_grad_cam < 0.04, torch.zeros(224, 224), saliency_grad_cam)
        print(f"\tPixels above 0: {count_nonzero(multiplier)}")
        for i, img in enumerate(input_batch*multiplier):
            save_img(img, "Grad-CAM", i+1)
    else:
        multiplier = 1.0

    plot_example(input_batch*multiplier, saliency_grad_cam, "Grad-CAM",
                category_id, save_path=f"output_{NET}_Grad-CAM.jpg")



    print("Processing Gradual Grad-CAM...")
    GradualExtrapolator.register_hooks(model)
    # we just need to process images to feed hooks
    model(input_batch)
    saliency_gradual_grad_cam = GradualExtrapolator.get_smooth_map(orig_saliency_grad_cam)



    # print(saliency_gradual_grad_cam[0].tolist())
    saliency_gradual_grad_cam = sum_pixels(saliency_gradual_grad_cam)
    if TRIM_SALIENCES:
        saliency_gradual_grad_cam = trim_map(saliency_gradual_grad_cam)

    if CLIP_INPUT:
        multiplier = torch.where(saliency_gradual_grad_cam < 0.07, torch.zeros(224, 224), saliency_gradual_grad_cam)
        print(f"\tPixels above 0: {count_nonzero(multiplier)}")
        for i, img in enumerate(input_batch*multiplier):
            save_img(img, "Gradual-Grad-CAM", i+1)
    else:
        multiplier = 1.0

    plot_example(input_batch*multiplier, saliency_gradual_grad_cam, "Gradual Grad-CAM",
            category_id, save_path=f"output_{NET}_Gradual-Grad-CAM.jpg")



    print("Processing Contrastive Excitation BP...")
    saliency_contr_excitation = contrastive_excitation_backprop(
        model,
        input_batch,
        category_id,
        saliency_layer=layers[NET][1],
        contrast_layer=layers[NET][1]
    )

    orig_saliency_contr_excitation = saliency_contr_excitation
    saliency_contr_excitation = sum_pixels(saliency_contr_excitation)

    if CLIP_INPUT:
        multiplier = torch.where(saliency_contr_excitation < 0.001, torch.zeros(224, 224), saliency_contr_excitation)
        print(f"\tPixels above 0: {count_nonzero(multiplier)}")
        for i, img in enumerate(input_batch*multiplier):
            save_img(img, "Contr-Excit-BP", i+1)
    else:
        multiplier = 1.0

    plot_example(input_batch*multiplier, saliency_contr_excitation, "Contrastive Excitation BP",
            category_id, save_path=f"output_{NET}_Contr-Excit-BP.jpg")

    print("Processing Gradual Contrastive Excitation BP...")
    GradualExtrapolator.reset_excitations()
    # we just need to process images to feed hooks
    model(input_batch)
    saliency_gradual_contr_excitation = GradualExtrapolator.get_smooth_map(orig_saliency_contr_excitation)

    # saliency_gradual_contr_excitation = sum_pixels(saliency_gradual_contr_excitation)

    if TRIM_SALIENCES:
        saliency_gradual_contr_excitation = trim_map(saliency_gradual_contr_excitation)

    if CLIP_INPUT:
        multiplier = torch.where(saliency_gradual_contr_excitation < 0.00005, torch.zeros(224, 224), saliency_gradual_contr_excitation)
        print(f"\tPixels above 0: {count_nonzero(multiplier)}")
        for i, img in enumerate(input_batch*multiplier):
            save_img(img, "Gradual-Contr-Excit-BP", i+1)
    else:
        multiplier = 1.0

    plot_example(input_batch*multiplier, saliency_gradual_contr_excitation, "Gradual Contrastive Excitation BP",
                category_id, save_path=f"output_{NET}_Gradual-Contr-Excit-BP.jpg")
