import rootutils
import numpy as np
import onnxruntime as ort

rootutils.setup_root(
    __file__, indicator=['.git', 'pyproject.toml'], pythonpath=True
)

from notebooks.helpers.inference_base import InferenceBase

class InferenceOnnx(InferenceBase):
    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        self.model = self.load_model()

    def load_model(self) -> None:
        model = ort.InferenceSession(
            self._model_path,
            providers=['CUDAExecutionProvider']
        )
        return model
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        input_name = self.model.get_inputs()[0].name
        output_names = [output.name for output in self.model.get_outputs()]
        outputs = self.model.run(output_names, {input_name: data})

        return outputs