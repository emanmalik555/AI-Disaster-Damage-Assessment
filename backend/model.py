import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


MODEL_PATH = "backend/best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    print("Building model...")

    model = smp.Unet(
        encoder_name="mit_b3",
        encoder_weights=None,
        in_channels=3,
        classes=5
    )

    # The trained model expects 6 input channels.
    # We replace the first convolution with a 6-channel version.
    original = model.encoder.patch_embed1.proj

    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=(original.bias is not None)
    )

    # Copy the original RGB weights into both RGB halves.
    with torch.no_grad():
        new_conv.weight[:, :3] = original.weight
        new_conv.weight[:, 3:] = original.weight

        if original.bias is not None:
            new_conv.bias.copy_(original.bias)

    model.encoder.patch_embed1.proj = new_conv

    # Load trained weights
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully on {DEVICE}")

    return model


model = build_model()