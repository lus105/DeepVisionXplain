from .preproc_strategy import PreprocessingStep

class PreprocessingPipeline:
    def __init__(self, steps: list[PreprocessingStep], overwrite: bool):
        """
        Initialize the pipeline with a list of preprocessing steps.
        
        Args:
            steps (list[PreprocessingStep]): List of PreprocessingStep instances.
            overwrite (bool): If true and data already preprocessed, overwrites it.
        """
        self.steps = steps
        self.overwrite = overwrite

    def run(self, data: dict) -> dict:
        """
        Execute all preprocessing steps in sequence.
        
        Args:
            data (dict): Initial data dictionary.
            overwrite_data (bool): if true, pipeline overwrites already preprocessed data.
        Returns:
            dict: Final data dictionary after all processing steps.
        """
        for step in self.steps:
            data = step.process(data, self.overwrite)
        return data