import pandas as pd
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from src.api.training.schemas import (
    MetricsResponse,
    MetricsErrorResponse,
    RunInfo,
    RunSummary,
)


class MetricsTracker:
    def __init__(self, logs_base_path: str = 'logs/train/runs'):
        """
        Initializes the object with a base path for storing log files.

        Args:
            logs_base_path (str, optional): The base directory path where training
                logs will be saved. Defaults to 'logs/train/runs'.
        """
        self.logs_base_path = Path(logs_base_path)

    def get_latest_run_metrics(self) -> Union[MetricsResponse, MetricsErrorResponse]:
        """
        Retrieves the metrics for the most recent training run.
        Returns:
            MetricsResponse: The metrics of the latest training run if available.
            MetricsErrorResponse: An error response if no training runs are found.
        """
        latest_run = self._find_latest_run()
        if not latest_run:
            return MetricsErrorResponse(error='No training runs found')

        return self.get_run_metrics(latest_run.name)

    def get_run_metrics(
        self, run_id: str
    ) -> Union[MetricsResponse, MetricsErrorResponse]:
        """
        Retrieves the metrics for a specific training run.
        Args:
            run_id (str): The unique identifier of the training run.
        Returns:
            Union[MetricsResponse, MetricsErrorResponse]: 
                - MetricsResponse containing parsed metrics if successful.
                - MetricsErrorResponse with an error message if the run or metrics CSV is not found.
        """
        run_path = self.logs_base_path / run_id
        if not run_path.exists():
            return MetricsErrorResponse(error=f'Run {run_id} not found')

        # Look for CSV files in the run directory
        csv_file = self._find_metrics_csv(run_path)
        if not csv_file:
            return MetricsErrorResponse(error=f'No metrics CSV found for run {run_id}')

        return self._parse_csv_metrics(csv_file, run_id)

    def list_available_runs(self) -> list[RunInfo]:
        """
        Retrieves a list of available training runs from the logs base directory.
        Iterates through each subdirectory in the logs base path, collects run information
        using the `_get_run_info` method, and returns a list of `RunInfo` objects sorted
        by creation time in descending order (newest first).
        Returns:
            list[RunInfo]: A list of `RunInfo` objects representing available runs,
                sorted by creation time (newest first). Returns an empty list if the logs
                base path does not exist.
        """
        if not self.logs_base_path.exists():
            return []

        runs = []
        for run_dir in self.logs_base_path.iterdir():
            if run_dir.is_dir():
                run_info = self._get_run_info(run_dir)
                if run_info:
                    runs.append(run_info)

        # Sort by creation time, newest first
        return sorted(runs, key=lambda x: x.created_at, reverse=True)

    def get_run_summary(self, run_id: str) -> Union[RunSummary, MetricsErrorResponse]:
        """
        Generate a summary of training run metrics, including best and final values.
        Args:
            run_id (str): The unique identifier for the training run.
        Returns:
            Union[RunSummary, MetricsErrorResponse]: A RunSummary object containing
                the run's best and final metrics, or a MetricsErrorResponse if metrics
                retrieval fails.
        """
        metrics = self.get_run_metrics(run_id)
        if isinstance(metrics, MetricsErrorResponse):
            return metrics

        # Calculate best and final metrics
        best_metrics = {}
        final_metrics = {}

        if metrics.data:
            # Final metrics (last row)
            final_row = metrics.data[-1]
            final_metrics = {k: v for k, v in final_row.items() if v is not None}

            # Best metrics (find best values for each metric)
            for col in metrics.available_columns:
                if col in ['epoch', 'step']:
                    continue

                values = [
                    row.get(col) for row in metrics.data if row.get(col) is not None
                ]
                if values:
                    if 'loss' in col.lower():
                        best_metrics[f'best_{col}'] = min(values)
                    elif 'acc' in col.lower():
                        best_metrics[f'best_{col}'] = max(values)

        return RunSummary(
            run_id=run_id,
            total_epochs=metrics.max_epoch,
            total_steps=metrics.max_step,
            best_metrics=best_metrics,
            final_metrics=final_metrics,
            available_columns=metrics.available_columns,
        )

    def _find_latest_run(self) -> Optional[Path]:
        """
        Finds the most recently modified run directory within the logs base path.
        Returns:
            Optional[Path]: The path to the most recently modified run directory,
                or None if the logs base path does not exist or contains no directories.
        """
        if not self.logs_base_path.exists():
            return None

        run_dirs = [d for d in self.logs_base_path.iterdir() if d.is_dir()]
        if not run_dirs:
            return None

        # Sort by modification time, most recent first
        return max(run_dirs, key=lambda d: d.stat().st_mtime)

    def _find_metrics_csv(self, run_path: Path) -> Optional[Path]:
        """
        Searches for a metrics CSV file within the specified run directory using
        common Lightning log patterns.
        Args:
            run_path (Path): The root directory to search for metrics CSV files.
        Returns:
            Optional[Path]: The path to the most recently modified metrics CSV
                file if found, otherwise None.
        """
        # Common patterns for Lightning CSV logs
        patterns = [
            'csv/version_*/metrics.csv',
            'csv/metrics.csv',
            'lightning_logs/version_*/metrics.csv',
            'metrics.csv',
        ]

        for pattern in patterns:
            csv_files = list(run_path.glob(pattern))
            if csv_files:
                # Return the most recent version if multiple exist
                return max(csv_files, key=lambda f: f.stat().st_mtime)

        return None

    def _parse_csv_metrics(
        self, csv_file: Path, run_id: str
    ) -> Union[MetricsResponse, MetricsErrorResponse]:
        """
        Parses a CSV file containing training metrics and returns a structured response.
        Args:
            csv_file (Path): Path to the CSV file containing metrics data.
            run_id (str): Identifier for the training run.
        Returns:
            Union[MetricsResponse, MetricsErrorResponse]: 
                - MetricsResponse: If the CSV is successfully parsed, containing run ID,
                available columns, total rows, max epoch, max step, parsed data,
                    CSV file path, and last modified timestamp.
                - MetricsErrorResponse: If parsing fails, containing an error message and the CSV file path.
        """
        try:
            df = pd.read_csv(csv_file)

            # Get available columns
            available_columns = list(df.columns)

            # Convert DataFrame to list of dictionaries for JSON serialization
            data = []
            for _, row in df.iterrows():
                row_dict = {}
                for col in available_columns:
                    value = row[col]
                    # Handle NaN values
                    if pd.isna(value):
                        row_dict[col] = None
                    else:
                        row_dict[col] = (
                            float(value) if isinstance(value, (int, float)) else value
                        )
                data.append(row_dict)

            # Calculate some basic statistics
            max_epoch = df['epoch'].max() if 'epoch' in df.columns else 0
            max_step = df['step'].max() if 'step' in df.columns else 0

            return MetricsResponse(
                run_id=run_id,
                available_columns=available_columns,
                total_rows=len(df),
                max_epoch=int(max_epoch) if not pd.isna(max_epoch) else 0,
                max_step=int(max_step) if not pd.isna(max_step) else 0,
                data=data,
                csv_file=str(csv_file),
                last_modified=datetime.fromtimestamp(
                    csv_file.stat().st_mtime
                ).isoformat(),
            )

        except Exception as e:
            return MetricsErrorResponse(
                error=f'Failed to parse CSV: {str(e)}', csv_file=str(csv_file)
            )

    def _get_run_info(self, run_dir: Path) -> Optional[RunInfo]:
        """
        Retrieves information about a training run located in the specified directory.
        Args:
            run_dir (Path): The path to the directory containing the run data.
        Returns:
            Optional[RunInfo]: An instance of RunInfo containing details about the run,
                such as run ID, creation and modification timestamps, status, presence of metrics,
                and the path to the metrics file if available. Returns None if any error occurs
                during retrieval.
        """
        try:
            stat = run_dir.stat()

            # Check if there's a metrics file
            csv_file = self._find_metrics_csv(run_dir)
            has_metrics = csv_file is not None

            # Determine status based on file presence
            status = 'completed' if has_metrics else 'in_progress'

            return RunInfo(
                run_id=run_dir.name,
                created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                status=status,
                has_metrics=has_metrics,
                metrics_file=str(csv_file) if csv_file else None,
            )
        except Exception:
            return None
