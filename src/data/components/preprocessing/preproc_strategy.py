from abc import ABC, abstractmethod

class PreprocessingStep(ABC):
    @abstractmethod
    def process(self, data: dict) -> dict:
        """
        Process the data and return the modified data directories.
        
        Args:
            data (dict): Input data dictionary containing input directories.
        
        Returns:
            dict: Modified data dictionary of directories after processing.
        """
        pass