import argparse
import numpy as np
import onnxruntime as ort


def load_and_run_onnx_model(onnx_file_path, input_size=(1, 3, 224, 224)):
    # Load the ONNX model
    session = ort.InferenceSession(onnx_file_path)

    # Generate a dummy input for the model. Adjust the size as needed.
    dummy_input = np.random.randn(*input_size).astype(np.float32)

    # Get the name of the input and output nodes
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # Run inference
    outputs = session.run(output_names, {input_name: dummy_input})

    # Output could be a list of tensors, depending on the model's output
    # For demonstration, we're just printing the outputs
    for i, output in enumerate(outputs):
        print(f"Output {i}: Shape: {output.shape}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load and run an ONNX model")
    parser.add_argument("--onnx_file_path", help="Path to the ONNX model file")
    parser.add_argument(
        "--input_size",
        nargs="+",
        type=int,
        default=[1, 3, 224, 224],
        help="Input size of the model",
    )
    args = parser.parse_args()

    load_and_run_onnx_model(args.onnx_file_path, args.input_size)


if __name__ == "__main__":
    main()
