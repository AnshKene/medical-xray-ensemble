import torch
import torch.nn.functional as F
import cv2
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam(model, image_tensor, target_layer, save_path):
    """
    Generate a Grad-CAM heatmap overlay and save it to save_path.

    Args:
        model        : trained PyTorch model (must be in eval mode)
        image_tensor : preprocessed tensor of shape (1, C, H, W)
        target_layer : the convolutional layer to hook (e.g. model.layer4)
        save_path    : file path to write the output PNG

    Returns:
        pred_class (int) — the predicted class index

    Note on frozen models:
        pytorch_grad_cam's gradient hook skips layers whose output has
        requires_grad=False.  Since the backbone is frozen, we temporarily
        re-enable grad on the target layer's parameters so the hook fires,
        then restore the frozen state afterwards.
    """
    model.eval()

    # ── Step 1: Get predicted class (no_grad is safe here) ───────────────────
    with torch.no_grad():
        outputs = model(image_tensor)

    if outputs.ndim != 2:
        raise ValueError(f"Expected 2-D output (batch, classes), got: {outputs.shape}")

    pred_class = int(outputs.argmax(dim=1).item())
    targets    = [ClassifierOutputTarget(pred_class)]

    # ── Step 2: Temporarily unfreeze target layer so grad hook fires ─────────
    # GradCAM's ActivationsAndGradients.save_gradient() early-returns when
    # output.requires_grad is False.  Enabling grad on the target layer's
    # params makes the forward output require_grad, so the hook captures it.
    frozen_params = []
    for p in target_layer.parameters():
        if not p.requires_grad:
            frozen_params.append(p)
            p.requires_grad_(True)

    try:
        # ── Step 3: Grad-CAM ──────────────────────────────────────────────────
        cam           = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
        cam.activations_and_grads.release()   # clean up hooks
    finally:
        # ── Step 4: Restore frozen state ──────────────────────────────────────
        for p in frozen_params:
            p.requires_grad_(False)

    # ── Step 5: Build overlay and save ────────────────────────────────────────
    img = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = img.astype(np.float32)          # show_cam_on_image requires float32

    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return pred_class