from components.vit_rollout_multihead import VitRolloutMultihead
from components.cnn_cam_multihead import CNNCAMMultihead

import argparse
import torch
import sys
from pathlib import Path

# Add the parent directory to sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
print(parent_dir)


def convert_to_onnx(
    model: str, input_size: tuple = (1, 3, 224, 224), onnx_file_path: str = "model.onnx"
):
    if model == "vit":
        model_instance = VitRolloutMultihead(visualize=True)
    elif model == "cnn":
        model_instance = CNNCAMMultihead()
    else:
        raise ValueError(f"Unsupported model type: {model}")

    # Dummy input corresponding to the input size of the model
    x = torch.randn(*input_size)
    # Export the model
    torch.onnx.export(
        model_instance,
        x,
        onnx_file_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["features", "output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "features": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Convert to an ONNX model")
    parser.add_argument(
        "--model", help="model to convert to ONNX", choices=["vit", "cnn"]
    )
    parser.add_argument(
        "--input_size",
        nargs="+",
        type=int,
        default=[1, 3, 224, 224],
        help="Input size of the model",
    )
    parser.add_argument(
        "--onnx_file_path",
        default="trained_models/model.onnx",
        help="Path to the ONNX model file",
    )
    args = parser.parse_args()

    # Convert input_size argument to a tuple
    input_size = tuple(args.input_size)

    convert_to_onnx(args.model, input_size, args.onnx_file_path)


if __name__ == "__main__":
    main()
