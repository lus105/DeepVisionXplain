import rootutils
import numpy as np
import onnxruntime as ort

rootutils.setup_root(
    __file__, indicator=['.git', 'pyproject.toml'], pythonpath=True
)

from notebooks.helpers.inference_base import InferenceBase

class InferenceOnnx(InferenceBase):
    """
    Inference implementation for ONNX models using ONNX Runtime.
    """
    def initialize(self) -> None:
        try:
            self.model = ort.InferenceSession(
                self._model_path,
                providers=["CUDAExecutionProvider"]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def predict(self, data: np.ndarray) -> np.ndarray:
        try:
            input_name = self.model.get_inputs()[0].name
            output_names = [output.name for output in self.model.get_outputs()]
            outputs = self.model.run(output_names, {input_name: data})
            
            return outputs
        except Exception as e:
            raise RuntimeError(f"ONNX prediction failed: {e}")