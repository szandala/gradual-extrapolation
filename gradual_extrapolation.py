from torch.nn import Conv2d, MaxPool2d
from torch import no_grad
from torch.nn.functional import interpolate


class GradualExtrapolator:
    _excitations = []
    _hook_handlers = []
    _is_orig_image = True

    def _excitation_hook(module, input, output):
        # for better output sharpness we collect input images
        if GradualExtrapolator._is_orig_image:
            GradualExtrapolator._excitations.append(input[0])
            GradualExtrapolator._is_orig_image = False
        GradualExtrapolator._excitations.append(output)

    def register_hooks(model, recursive=False):
        if not recursive and GradualExtrapolator._hook_handlers:
            print("Hooks can only be registered to one model at once. Please use: `prune_old_hooks()`")
            return

        first_layer = True
        for i, layer in enumerate(model.children()):
            if list(layer.children()):
                GradualExtrapolator.register_hooks(layer, recursive=True)
            elif isinstance(layer, MaxPool2d):
                GradualExtrapolator._hook_handlers.append(
                    layer.register_forward_hook(GradualExtrapolator._excitation_hook)
                )
            elif isinstance(layer, Conv2d) and layer.stride > (1, 1) and first_layer:
                GradualExtrapolator._hook_handlers.append(
                    layer.register_forward_hook(GradualExtrapolator._excitation_hook)
                )
                first_layer = False
            # elif isinstance(layer, Conv2d) and layer.stride > (1, 1):
            #     GradualExtrapolator._hook_handlers.append(
            #         layer.register_forward_hook(GradualExtrapolator._excitation_hook)
            #     )

    def prune_old_hooks(model):
        if not GradualExtrapolator._hook_handlers:
            print("No hooks to remove")
        for hook in GradualExtrapolator._hook_handlers:
            hook.remove()

        GradualExtrapolator._hook_handlers = []

    ###############################################

    def _upsampling(heatmap, pre_excitations):


        for e in pre_excitations[::-1]:
            # print("\n--------------")
            # print(f"Origin shape: {heatmap.shape}")
            # print(f"Using shape: {e.shape}")
            if heatmap.shape[-1] >= e.shape[-1]:
                # print("skipping")
                continue
            heatmap = interpolate(
                heatmap,
                size=(e.shape[2], e.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            heatmap *= (e.sum(dim=1, keepdim=True)/e.max())
        return heatmap

    def _normalize_to_rgb(features):
        scaled_features = (features - features.mean()) / features.std()
        scaled_features = scaled_features.clip(-1, 1)
        scaled_features = (scaled_features - scaled_features.min()) / (
            scaled_features.max() - scaled_features.min()
        )
        return scaled_features

    def get_smooth_map(heatmap):
        if not GradualExtrapolator._excitations:
            print("No data in hooks. Have You used `register_hooks(model)` method?")
            return

        print(f"Searched: {heatmap.shape}")
        [print(e.shape) for e in GradualExtrapolator._excitations]
        # # discarding further layers
        # new_excitations = []
        # for e in GradualExtrapolator._excitations:
        #     if e.shape[2:] == heatmap.shape[2:]:
        #         break
        #     new_excitations.append(e)
        # GradualExtrapolator._excitations = new_excitations
        # print("New excitations:")
        # [print(e.shape) for e in GradualExtrapolator._excitations]

        with no_grad():
            heatmap = GradualExtrapolator._upsampling(
                heatmap, GradualExtrapolator._excitations
            )
            rgb_features_map = GradualExtrapolator._normalize_to_rgb(heatmap)

            # prune old GradualExtrapolator._excitations
            GradualExtrapolator.reset_excitations()

            return rgb_features_map

    def reset_excitations():
        GradualExtrapolator._is_orig_image = True
        GradualExtrapolator._excitations = []
