class PreprocessingPipeline:
    def __init__(self, steps: list):
        """
        Initialize the pipeline with a list of preprocessing steps.
        
        Args:
            steps (list): List of PreprocessingStep instances.
        """
        self.steps = steps

    def run(self, data: dict, overwrite: bool) -> dict:
        """
        Execute all preprocessing steps in sequence.
        
        Args:
            data (dict): Initial data dictionary.
            overwrite_data (bool): if true, pipeline overwrites already preprocessed data.
        Returns:
            dict: Final data dictionary after all processing steps.
        """
        for step in self.steps:
            data = step.process(data, overwrite)
        return data