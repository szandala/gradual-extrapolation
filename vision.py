from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.excitation_backprop import contrastive_excitation_backprop
from torchray.attribution.deconvnet import deconvnet
from torchray.attribution.guided_backprop import guided_backprop

from utils import read_images_2_batch, prepare_network, get_category_IDs
from gradual_extrapolation import GradualExtrapolation


if __name__ == "__main__":
    input_batch = read_images_2_batch()
    model = prepare_network("vgg16")
    category_id = get_category_IDs(model, input_batch)
    print("Processing DeConvNet...")
    saliency = deconvnet(model, input_batch, category_id)
    # plot_example(input_batch, saliency, "DeConvNet",
    #             category_id, save_path="output_deconvnet.jpg")

    print("Processing Grad-CAM...")


    saliency_grad_cam = grad_cam(
        model,
        input_batch,
        category_id,
        saliency_layer="features.28",
    )
    plot_example(input_batch, saliency_grad_cam, "Grad-CAM",
                category_id, save_path="output_grad-cam.jpg")



    print("Processing Gradual Grad-CAM...")
    SmoothExtrapolator.register_hooks(model)
    category_id = get_category_n_percentage(model, input_batch)


    saliency_smooth_grad_cam = SmoothExtrapolator.get_smooth_map(saliency_grad_cam)

    # ##########################
    # # plt.clf()
    # # plt.hist(saliency_smooth_grad_cam[0][0], bins = 256)
    # # plt.savefig("histogram.png", format="png", dpi=500) #, bbox_inches="tight")

    # ##########################



    # s_min = saliency_smooth_grad_cam.min()
    # s_max = saliency_smooth_grad_cam.max()
    # print(f"min = {s_min}, max = {s_max}")
    # saliency_smooth_grad_cam = (saliency_smooth_grad_cam - 0.1).clip(0, 1)
    # saliency_smooth_grad_cam = (saliency_smooth_grad_cam - saliency_smooth_grad_cam.min())/(saliency_smooth_grad_cam.max() - saliency_smooth_grad_cam.min())

    # ##########################
    # # plt.clf()
    # # plt.hist(saliency_smooth_grad_cam[0][0], bins = 256)
    # # plt.savefig("histogram2.png", format="png", dpi=500) #, bbox_inches="tight")

    # ##########################

    plot_example(input_batch, saliency_smooth_grad_cam, "Gradual Grad-CAM",
            category_id, save_path="output_Gradual-Grad-CAM.jpg")

    # saliency_prism = saliency

    # category_id = [ c - 100 for c in category_id]









##################################
    # print("Processing Grad-CAMs...")

    # grad_CAMs = []
    # layers = [
    #     # "features.0",
    #     # "features.8",
    #     # "features.15",
    #     "features.16",
    #     "features.17",
    #     "features.18",
    #     "features.19",
    #     # "features.20",
    #     # "features.21",
    #     # "features.22",
    #     "features.28"
    # ]
    # for layer in layers:
    #     saliency_grad_cam = grad_cam(
    #         model,
    #         input_batch,
    #         category_id,
    #         saliency_layer=layer,
    #     )

    #     grad_CAMs.append(saliency_grad_cam.squeeze(0))

    # for i,l in enumerate(grad_CAMs):
    #     print(f"{layers[i]} = {l.shape}")

    # def extrapolate(saliencies):
    #     heatmap = saliencies.pop()
    #     alles = []
    #     for saliency in saliencies:
    #         heatmap = interpolate(
    #             heatmap,
    #             size=(saliency.shape[2], saliency.shape[3]),
    #             mode="bilinear",
    #             align_corners=False,
    #         )
    #         heatmap *= saliency
    #         alles.append(heatmap.copy())
    #     return heatmap, alles

    #     plt.figure(figsize=(2,3)) # specifying the overall grid size

    # for i in range(len(grad_CAMs)):
    #     plt.subplot(2,3,i+1)    # the number of images in the grid is 5*5 (25)
    #     bitmap = grad_CAMs[i].expand(*grad_CAMs[i].shape).permute(1, 2, 0).cpu().detach().numpy()
    #     handle = plt.imshow(
    #         bitmap, cmap="jet", interpolation="bilinear")

    # plt.savefig("output_gradual-grad-cam.jpg", format="jpg", dpi=500)
    # plot_example(input_batch, extrapolate(grad_CAMs), "Gradual-Grad-CAM",
    #     category_id, save_path="output_gradual-grad-cam.jpg")
    # print("Processing Smooth Grad-CAM...")
    # SmoothExtrapolator.register_hooks(model)
    # category_id = get_category_n_percentage(model, input_batch)


    # saliency_smooth_grad_cam = SmoothExtrapolator.get_smooth_map(saliency_grad_cam)












    print("Processing Gradual Contrastive Excitation BP...")
    # SmoothExtrapolator.register_hooks(model)
    category_id = get_category_n_percentage(model, input_batch)


    saliency_smooth_excitation = contrastive_excitation_backprop(
        model,
        input_batch,
        category_id,
        saliency_layer="features.18",
        contrast_layer="features.18"
    )


    saliency_smooth_excitation = SmoothExtrapolator.get_smooth_map(saliency_smooth_excitation)

    plot_example(input_batch, saliency_smooth_excitation, "Gradual Excitation BP",
            category_id, save_path="output_Gradual-Excitation.jpg")

    print("Processing Contrastive Excitation BP...")
    saliency = contrastive_excitation_backprop(
        model,
        input_batch,
        category_id,
        saliency_layer="features.18",
        contrast_layer="features.18"
    )
    plot_example(input_batch, saliency, "Excitation BP",
                category_id, save_path="output_excitationBP.jpg")


    # print("Processing gradient..")
    # saliency = gradient(model, input_batch, category_id)
    # plot_example(input_batch, saliency, "Gradient",
    #             category_id, save_path="output_gradient.jpg")

    # for name, param in model.named_parameters():
    #     print(name)

    # print("Processing Guided BP...")
    # saliency = guided_backprop(model, input_batch, category_id)
    # plot_example(input_batch, saliency, "Guided BP",
    #             category_id, save_path="output_guidedBP.jpg")
    # saliency_guided_bp = saliency

    # print("Processing Guided Grad-CAM...")
    # saliency_grad_cam = interpolate(saliency, size=saliency_guided_bp[0].shape[1:], mode="bilinear", align_corners=False)
    # plot_example(input_batch, saliency_guided_bp * saliency_grad_cam, "Guided Grad-CAM",
    #             category_id, save_path="output_guided-grad-cam.jpg")

    # print("Processing Linear-Approx...")
    # saliency = linear_approx(
    #     model,
    #     input_batch,
    #     category_id,
    #     saliency_layer="features",
    # )
    # plot_example(input_batch, saliency, "Linear-Approx",
    #             category_id, save_path="output_linear-approx.jpg")

    # # print("Processing RISE...")
    # # saliency = rise(model, input_batch, num_masks=100)
    # # print(f"X {saliency.shape}")
    # # saliency = saliency[:, category_id]
    # # print(f"Y {saliency.shape}")
    # # plot_example(input_batch, saliency, "RISE",
    # #         category_id, save_path="output_RISE.jpg")
