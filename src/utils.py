from collections import OrderedDict
import torch


def conversion(path: str) -> OrderedDict:
    try:
        # It's good practice to load to the CPU first
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(
            "Please replace 'path/to/your/pretrained_weights.pth' with the actual file path."
        )
        state_dict = {}  # Assign empty dict to prevent further errors in this example

    # The weights might be inside a 'model' key in the checkpoint file
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # 3. Create a new state_dict with corrected key names
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Skip layers from the pre-training heads
        if "mask_token" in key:
            continue

        new_key = key

        # Remove the 'student.' prefix
        if new_key.startswith("student."):
            new_key = new_key.replace("student.", "")

            # Rename specific layers to match braindecode's implementation
            if new_key == "pos_embed":
                new_key = "position_embedding"
            elif new_key == "time_embed":
                # new_key = 'temporal_embedding'
                continue
            elif "patch_embed.conv" in new_key:
                new_key = new_key.replace(
                    "patch_embed.conv", "patch_embed.temporal_conv.conv"
                )
            elif "patch_embed.norm" in new_key:
                new_key = new_key.replace(
                    "patch_embed.norm", "patch_embed.temporal_conv.norm"
                )
            elif "mlp.fc1" in new_key:
                new_key = new_key.replace("mlp.fc1", "mlp.0")
            elif "mlp.fc2" in new_key:
                new_key = new_key.replace("mlp.fc2", "mlp.2")
            elif "lm_head" in new_key:
                new_key = new_key.replace("lm_head", "final_layer")

            # Ignore attention norm layers not present in braindecode's model
            if "attn.q_norm" in new_key or "attn.k_norm" in new_key:
                continue

            new_state_dict[new_key] = value

    return new_state_dict
