from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copy2
from .preproc_strategy import PreprocessingStep
from ..utils import (DatasetType,
                     list_files,
                     list_dirs,
                     find_annotation_file,
                     clear_directory,
                     IMAGE_EXTENSIONS,
                     XML_EXTENSION,
                     JSON_EXTENSION)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SplitStep(PreprocessingStep):
    def __init__(
        self,
        split_ratio: tuple,
        seed: int,
    ):
        super().__init__()
        self.split_ratio = split_ratio
        self.seed = seed

    def process(self, data: dict, overwrite: bool) -> dict:
        if not round(sum(self.split_ratio), 5) == 1:
            raise ValueError('The sums of `ratio` is over 1.')
        data_path = Path(data['initial_data'])
        base_path = data_path.parent
        last_subdir = data_path.name
        output_path = base_path / f'{last_subdir}_processed'

        if (output_path.exists() and overwrite):
            clear_directory(output_path)
            output_path.rmdir()
        elif (output_path.exists() and not overwrite):
            log.info('Data is already splitted. In order to overwrite, set data.overwrite=True')
            return
        
        dataset_type = self._determine_dataset_type(data_path)
        data_frame = self._to_dataframe(data_path, dataset_type)
        data_frame_splitted = self._split_dataset(data_frame)
        self._save_split(data_frame_splitted, output_path, dataset_type)
        print('done')

    def _determine_dataset_type(self, data_path: Path) -> DatasetType:
        def count_class_folders(path: Path) -> int:
            return sum(1 for item in path.iterdir() if item.is_dir())

        if (data_path / self._image_subdir).exists() and (data_path / self._label_subdir).exists():
            image_folder_count = count_class_folders(data_path / self._image_subdir)
            
            if image_folder_count >= 2:
                return DatasetType.ImageLabelMultiClass
            elif image_folder_count == 0:
                return DatasetType.ImageLabelBinary
            else:
                raise ValueError('Unknown dataset structure in image/label subdirectories')

        root_folder_count = count_class_folders(data_path)
        
        if root_folder_count >= 3:
            return DatasetType.ImageMultiClass
        elif root_folder_count == 2:
            return DatasetType.ImageBinary

        raise ValueError('Unknown dataset structure')
    
    def _to_dataframe(self, data_path: Path, dataset_type: DatasetType) -> pd.DataFrame:
        data = []

        if dataset_type == DatasetType.ImageLabelBinary:
            image_dir = data_path / self._image_subdir
            label_dir = data_path / self._label_subdir
            image_files = list_files(image_dir, IMAGE_EXTENSIONS)
            
            for image_path in image_files:
                label_path = find_annotation_file(
                    label_dir,
                    image_path.stem,
                    IMAGE_EXTENSIONS + [XML_EXTENSION, JSON_EXTENSION]
                )
                if label_path:
                    data.append({
                        'image_path': image_path,
                        'label_path': label_path,
                        'class': "binary"
                    })

        elif dataset_type == DatasetType.ImageLabelMultiClass:
            image_dir = data_path / self._image_subdir
            label_dir = data_path / self._label_subdir
            class_dirs = list_dirs(image_dir)
            
            for class_dir in class_dirs:
                class_name = class_dir.name
                image_files = list_files(class_dir, IMAGE_EXTENSIONS)
                
                for image_path in image_files:
                    label_path = find_annotation_file(
                        label_dir / class_name,
                        image_path.stem,
                        IMAGE_EXTENSIONS + [XML_EXTENSION, JSON_EXTENSION]
                    )
                    if label_path:
                        data.append({
                            'image_path': image_path,
                            'label_path': label_path,
                            'class': class_name
                        })

        elif dataset_type == DatasetType.ImageBinary:
            class_dirs = list_dirs(data_path)
            
            for class_dir in class_dirs:
                class_name = class_dir.name
                image_files = list_files(class_dir, IMAGE_EXTENSIONS)
                
                for image_path in image_files:
                    data.append({
                        'image_path': image_path,
                        'label_path': None,
                        'class': class_name
                    })

        elif dataset_type == DatasetType.ImageMultiClass:
            class_dirs = list_dirs(data_path)
            
            for class_dir in class_dirs:
                class_name = class_dir.name
                image_files = list_files(class_dir, IMAGE_EXTENSIONS)
                
                for image_path in image_files:
                    data.append({
                        'image_path': image_path,
                        'label_path': None,
                        'class': class_name
                    })

        else:
            raise ValueError('Unsupported dataset type')

        df = pd.DataFrame(data)
        return df

    def _split_dataset(self, df: pd.DataFrame)-> pd.DataFrame:
        train_size, test_size, val_size = self.split_ratio

        if 'class' not in df.columns:
            raise ValueError('DataFrame must contain a \'class\' column for stratified splitting.')

        num_classes = df['class'].nunique()
        total_samples = len(df)

        min_samples_per_split = num_classes
        test_count = max(int(total_samples * test_size), min_samples_per_split)
        val_count = max(int(total_samples * val_size), min_samples_per_split)
        train_count = total_samples - test_count - val_count

        if train_count < num_classes:
            raise ValueError('Not enough samples to create stratified splits with the current split ratios and number of classes.')

        train_df, remaining_df = train_test_split(
            df, train_size=train_count, random_state=self.seed, stratify=df['class']
        )
        train_df = train_df.copy()
        split_cn = 'split'
        train_df[split_cn] = self._train_subdir

        test_df, val_df = train_test_split(
            remaining_df, train_size=test_count, random_state=self.seed, stratify=remaining_df['class']
        )
        test_df = test_df.copy()
        test_df[split_cn] = self._test_subdir
        val_df = val_df.copy()
        val_df[split_cn] = self._val_subdir

        final_df = pd.concat([train_df, test_df, val_df], ignore_index=True)
        
        return final_df
    
    def _save_split(
            self,
            dataframe: pd.DataFrame,
            output_path: Path,
            dataset_type: DatasetType
        ) -> None:
        """Saves images and labels to separate directories based on the split and class.

        Args:
            dataframe (pd.DataFrame): DataFrame with columns `image_path`, `label_path`, `class`, and `split`.
            output_dir (Path): The root directory where split folders (train, test, val) will be created.
        """
        for _, row in dataframe.iterrows():
            split = row['split']
            class_name = row['class']

            label_dir = None
            if dataset_type == DatasetType.ImageBinary:
                image_dir = output_path / split / class_name
            elif dataset_type == DatasetType.ImageLabelBinary:
                image_dir = output_path / split / self._image_subdir
                label_dir = output_path / split / self._label_subdir
            elif dataset_type == DatasetType.ImageMultiClass:
                image_dir = output_path / split / class_name
            elif dataset_type == DatasetType.ImageLabelMultiClass:
                image_dir = output_path / split / self._image_subdir / class_name
                label_dir = output_path / split / self._label_subdir / class_name

            image_dir.mkdir(parents=True, exist_ok=True)
            if label_dir:
                label_dir.mkdir(parents=True, exist_ok=True)

            image_path_rn = 'image_path'
            image_dest = image_dir / Path(row[image_path_rn]).name
            copy2(row[image_path_rn], image_dest)

            label_path_rn = 'label_path'
            if pd.notna(row[label_path_rn]):  # Check if the label path is not NaN
                label_dest = label_dir / Path(row[label_path_rn]).name
                copy2(row[label_path_rn], label_dest)