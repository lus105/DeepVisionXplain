import rootutils
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

rootutils.setup_root(__file__, indicator=['.git', 'pyproject.toml'], pythonpath=True)

from notebooks.helpers.inference_base import InferenceBase


# https://stackoverflow.com/questions/59280745/inference-with-tensorrt-engine-file-on-python
class InferenceTensorRT(InferenceBase):
    def initialize(self):
        try:
            self.logger = trt.Logger(trt.Logger.ERROR)
            self.runtime = trt.Runtime(self.logger)

            # Load model
            with open(self._model_path, 'rb') as f:
                self.model = self.runtime.deserialize_cuda_engine(f.read())

            self.context = self.model.create_execution_context()

            # allocate buffers
            self.inputs, self.outputs, self.bindings, self.stream = (
                self._allocate_buffers()
            )
        except Exception as e:
            raise RuntimeError(f'Failed to load TensorRT model: {e}')

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem, shape):
            # keeping track of addresses
            self.host = host_mem
            self.device = device_mem
            # keeping track of shape to un-flatten it later
            self.shape = shape

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(self.model.num_io_tensors):
            tensor_name = self.model.get_tensor_name(i)
            shape = self.model.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(self.model.get_tensor_dtype(tensor_name))

            # allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # append to the appropiate input/output list
            if self.model.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem, shape))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem, shape))

        return inputs, outputs, bindings, stream

    def __call__(self, data: np.ndarray) -> np.ndarray:
        # transfer input data to device
        np.copyto(self.inputs[0].host, data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # set tensor address
        for i in range(self.model.num_io_tensors):
            self.context.set_tensor_address(
                self.model.get_tensor_name(i), self.bindings[i]
            )

        # run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # transfer predictions back
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(
                self.outputs[i].host, self.outputs[i].device, self.stream
            )

        # synchronize the stream
        self.stream.synchronize()

        # un-flatten the outputs
        outputs = []
        for i in range(len(self.outputs)):
            output = self.outputs[i].host
            output = output.reshape(self.outputs[i].shape)
            outputs.append(output)

        return outputs
