from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.excitation_backprop import contrastive_excitation_backprop
from torchray.attribution.deconvnet import deconvnet

from utils import read_images_2_batch, get_category_IDs, plot_example
from gradual_extrapolation import GradualExtrapolator
from torchvision import models

import torch
from torch.nn.functional import interpolate

def sum_pixels(saliency_map):
    # print(saliency_map.shape)
    saliency_map = interpolate(
                saliency_map,
                size=(input_batch.shape[2], input_batch.shape[3]),
                mode="bilinear",
                # align_corners=False,
            )

    print(saliency_map.shape)
    print(saliency_map.sum(dim=[3,2]))
    print("===")
    print(saliency_map.mean(dim=[3,2]).tolist())
    # for img in saliency_map.tolist():

    #     s = 0
    #     for row in img[0]:

    #         for y in row:

    #             s+= y
    #     print(f"Sum is: {s}")
    return saliency_map

if __name__ == "__main__":
    CLIP_INPUT = True
    input_batch = read_images_2_batch()

    # TODO: loop over all common networks
    model = models.__dict__["vgg16"](pretrained=True)
    model.eval()
    category_id = get_category_IDs(model, input_batch)
    deconvnet(model, input_batch, category_id)

    print("Processing Grad-CAM...")
    saliency_grad_cam = grad_cam(
        model,
        input_batch,
        category_id,
        saliency_layer="features.28",
    )
    saliency_grad_cam = sum_pixels(saliency_grad_cam)
    if CLIP_INPUT:
            multiplier = torch.where(saliency_grad_cam < 0.05, torch.zeros(224, 224), saliency_grad_cam)
    else:
        multiplier = 1.0

    plot_example(input_batch*multiplier, saliency_grad_cam, "Grad-CAM",
                category_id, save_path="output_Grad-CAM.jpg")


    print("Processing Gradual Grad-CAM...")
    GradualExtrapolator.register_hooks(model)
    # we just need to process images to feed hooks
    model(input_batch)
    saliency_gradual_grad_cam = GradualExtrapolator.get_smooth_map(saliency_grad_cam)

    saliency_gradual_grad_cam = sum_pixels(saliency_gradual_grad_cam)

    if CLIP_INPUT:
        multiplier = torch.where(saliency_gradual_grad_cam < 0.1, torch.zeros(224, 224), saliency_gradual_grad_cam)
    else:
        multiplier = 1.0

    plot_example(input_batch*multiplier, saliency_gradual_grad_cam, "Gradual Grad-CAM",
            category_id, save_path="output_Gradual-Grad-CAM.jpg")

    print("Processing Contrastive Excitation BP...")
    saliency_contr_excitation = contrastive_excitation_backprop(
        model,
        input_batch,
        category_id,
        saliency_layer="features.18",
        contrast_layer="features.18"
    )

    saliency_contr_excitation = sum_pixels(saliency_contr_excitation)

    if CLIP_INPUT:
            multiplier = torch.where(saliency_contr_excitation < 0.005, torch.zeros(224, 224), saliency_contr_excitation)
    else:
        multiplier = 1.0

    plot_example(input_batch*multiplier, saliency_contr_excitation, "Contrastive Excitation BP",
            category_id, save_path="output_Contr-Excit-BP.jpg")

    print("Processing Gradual Contrastive Excitation BP...")
    # we just need to process images to feed hooks
    model(input_batch)
    saliency_gradual_contr_excitation = GradualExtrapolator.get_smooth_map(saliency_contr_excitation)

    saliency_gradual_contr_excitation = sum_pixels(saliency_gradual_contr_excitation)

    if CLIP_INPUT:
            multiplier = torch.where(saliency_gradual_contr_excitation < 0.001, torch.zeros(224, 224), saliency_gradual_contr_excitation)
            # multiplier = (saliency_gradual_contr_excitation - saliency_gradual_contr_excitation.mean(dim=[2,3])).clip(0.0, 1.0)
            # print(f"base={saliency_gradual_contr_excitation.shape}\nmean={saliency_gradual_contr_excitation.mean(dim=[2,3]).shape}\nmulti={multiplier.shape}")
    else:
        multiplier = 1.0

    plot_example(input_batch*multiplier, saliency_gradual_contr_excitation, "Gradual Contrastive Excitation BP",
                category_id, save_path="output_Gradual-Contr-Excit-BP.jpg")
