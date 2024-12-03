from .preproc_strategy import PreprocessingStep


class PreprocessingPipeline:
    def __init__(self, steps: list[PreprocessingStep]):
        """
        Initialize the pipeline with a list of preprocessing steps.

        Args:
            steps (list[PreprocessingStep]): List of PreprocessingStep instances.
        """
        self.steps = steps

    def run(self, data: dict) -> dict:
        """
        Execute all preprocessing steps in sequence.

        Args:
            data (dict): Initial data dictionary.
        Returns:
            dict: Final data dictionary after all processing steps.
        """
        for step in self.steps:
            data = step.process(data)
        return data

    def get_processed_data_path(self, data: dict) -> dict:
        for step in self.steps:
            data = step.get_processed_data_path(data)
        return data
