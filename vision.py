from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.excitation_backprop import contrastive_excitation_backprop
from torchray.attribution.deconvnet import deconvnet
from torchray.attribution.guided_backprop import guided_backprop

from utils import read_images_2_batch, get_category_IDs, plot_example
from gradual_extrapolation import GradualExtrapolator
from torchvision import models


if __name__ == "__main__":
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
    plot_example(input_batch, saliency_grad_cam, "Grad-CAM",
                category_id, save_path="output_Grad-CAM.jpg")


    print("Processing Gradual Grad-CAM...")
    GradualExtrapolator.register_hooks(model)
    # we just need to process images to feed hooks
    model(input_batch)
    saliency_gradual_grad_cam = GradualExtrapolator.get_smooth_map(saliency_grad_cam)

    plot_example(input_batch, saliency_gradual_grad_cam, "Gradual Grad-CAM",
            category_id, save_path="output_Gradual-Grad-CAM.jpg")

    print("Processing Contrastive Excitation BP...")
    saliency_contr_excitation = contrastive_excitation_backprop(
        model,
        input_batch,
        category_id,
        saliency_layer="features.18",
        contrast_layer="features.18"
    )
    plot_example(input_batch, saliency_contr_excitation, "Contrastive Excitation BP",
            category_id, save_path="output_Contr-Excit-BP.jpg")

    print("Processing Gradual Contrastive Excitation BP...")
    model(input_batch)
    saliency_gradual_contr_excitation = GradualExtrapolator.get_smooth_map(saliency_contr_excitation)
    plot_example(input_batch, saliency_gradual_contr_excitation, "Gradual Contrastive Excitation BP",
                category_id, save_path="output_Gradual-Contr-Excit-BP.jpg")
