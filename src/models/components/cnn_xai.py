import numpy as np
import torch
from torch import nn


class CNNXAI(nn.Module):
    """
    A class for explainability in CNN models that supports feature extraction and 
    heatmap generation for the last convolutional and linear layers.
    
    Args:
        model (nn.Module): The pre-trained CNN model.
        last_conv_layer (nn.Module): The last convolutional layer in the model.
    """
    
    def __init__(self, model: nn.Module, last_conv_layer: nn.Module):
        """
        Initializes the CNNXAI module.

        Args:
            model (nn.Module): The model for which feature extraction and explainability 
                               is to be implemented.
            last_conv_layer (nn.Module): The last convolutional layer in the model.
        """
        super(CNNXAI, self).__init__()
        
        self.__last_conv_layer: nn.Module = last_conv_layer
        self.model: nn.Module = model
        
        self.__conv_activations: torch.Tensor = None  # Stores the activations of the last conv layer
        self.__last_linear_layer: nn.Linear = self.__find_last_linear_layer(self.model)
        self.__hook_handle = None
        
        self.__hook()
        
    def __hook(self) -> None:
        """
        Registers a forward hook on the last convolutional layer to capture activations 
        during the forward pass.
        """
        def __forward_hook(module: nn.Module, 
                           input: torch.Tensor, 
                           output: torch.Tensor) -> None:
            self.__conv_activations = output
            
        self.__hook_handle = self.__last_conv_layer.register_forward_hook(__forward_hook)
        
    def __unhook(self):
        """Detach the forward hook."""
        if self.__hook_handle is not None:
            self.__hook_handle.remove()
            self.__hook_handle = None
        else:
            print("No hook to detach.")
        
    def __find_last_linear_layer(self, module: nn.Module) -> nn.Linear:
        """
        Recursively finds the last Linear layer in the model.

        Args:
            module (nn.Module): The module (or submodule) to search for the last Linear layer.

        Returns:
            nn.Linear: The last Linear layer found in the model, or None if no Linear layer exists.
        """
        last_linear: nn.Linear = None
        for layer in module.children():
            if isinstance(layer, nn.Linear):
                last_linear = layer  # Update if Linear layer is found
            else:
                last_linear_submodule = self.__find_last_linear_layer(layer)
                if last_linear_submodule:
                    last_linear = last_linear_submodule
        return last_linear
    
    def __process_conv_output(self, output: torch.Tensor) -> np.ndarray:
        """
        Processes the output of the convolutional layer into a suitable format for 
        heatmap generation.

        Args:
            output (torch.Tensor): The raw output tensor from the last convolutional layer.

        Returns:
            np.ndarray: The processed output, converted to a NumPy array and reshaped as necessary.
        """
        conv_output: torch.Tensor = torch.squeeze(output)
        features_count: int = max(conv_output.shape)
        if conv_output.shape[-1] != features_count:
            conv_output = torch.reshape(conv_output, (*output.shape[2:], -1))
        conv_output = conv_output.detach().numpy()
        conv_output = conv_output.astype(np.uint8)
        return conv_output
    
    def __process_heatmap(self, conv_output: np.ndarray, last_layer_pred_weights: np.ndarray) -> np.ndarray:
        """
        Generates a heatmap based on the processed convolutional output and the weights
        of the prediction from the last linear layer.

        Args:
            conv_output (np.ndarray): Processed output from the convolutional layer.
            last_layer_pred_weights (np.ndarray): Weights corresponding to the prediction 
                                                  from the last linear layer.

        Returns:
            np.ndarray: The generated heatmap, normalized and scaled to uint8.
        """
        heat_map: np.ndarray = last_layer_pred_weights.dot(conv_output.reshape((last_layer_pred_weights.shape[0], -1)))
        heat_map = heat_map.reshape(conv_output.shape[:2])
        heat_map = heat_map - np.min(heat_map)
        heat_map = heat_map / np.max(heat_map)
        heat_map = np.uint8(255 * heat_map)
        return heat_map
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        """
        Forward pass through the model. In addition to returning the prediction, it generates 
        the explainability heatmap based on the activations of the last convolutional layer and
        the weights of the last linear layer.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            tuple[torch.Tensor, np.ndarray]: A tuple containing:
                - The predicted class index (torch.Tensor).
                - The heatmap (np.ndarray) for explainability.
        """
        logits: torch.Tensor = self.model(x)
        pred: torch.Tensor = torch.softmax(logits, dim=-1)
        pred_argmax = torch.argmax(pred)
        last_conv_out = self.__conv_activations
        last_conv_output_processed = self.__process_conv_output(last_conv_out)
        last_layer_weights_for_pred = self.__last_linear_layer.weight[pred_argmax, :].detach().numpy()
        heatmap = self.__process_heatmap(last_conv_output_processed, last_layer_weights_for_pred)
        return pred_argmax, heatmap
