from pathlib import Path
import random
from shutil import copy2
from os import path
from .preproc_strategy import PreprocessingStep
from ..utils import (list_files,
                     list_dirs,
                     IMAGE_EXTENSIONS,
                     XML_EXTENSION,
                     JSON_EXTENSION)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class SplitStep(PreprocessingStep):
    def __init__(
        self,
        split_ratio: tuple,
        random_state: int = 42,
    ):
        super().__init__()
        self.split_ratio = split_ratio
        random.seed(random_state)

    def process(self, data: dict, overwrite_data: bool) -> dict:
        if not round(sum(self.split_ratio), 5) == 1:
            raise ValueError("The sums of `ratio` is over 1.")
        data_path = Path(data['initial_data'])

        self._check_input_format(data_path)

        for class_dir in list_dirs(data_path):
            self.split_class_dir_ratio(class_dir)

    def _check_input_format(self, input: Path):
        p_input = Path(input)
        if not p_input.exists():
            err_msg = f'The provided input folder "{input}" does not exists.'
            if not p_input.is_absolute():
                err_msg += f' Your relative path cannot be found from the current working directory "{Path.cwd()}".'
            log.error(err_msg)
            raise ValueError(err_msg)

        if not p_input.is_dir():
            err_msg = f'The provided input folder "{input}" is not a directory'
            log.error(err_msg)
            raise ValueError(err_msg)

        dirs = list_dirs(input)
        if len(dirs) == 0:
            err_msg = f'The input data is not in a right format.'
            log.error(err_msg)
            raise ValueError(err_msg)
        
    def _split_class_dir_ratio(self, class_dir):
        files = self._setup_files(class_dir)
        split_train_idx = int(self.split_ratio[0] * len(files))
        split_val_idx = split_train_idx + int(self.split_ratio[1] * len(files))

        li = self._split_files(files, split_train_idx, split_val_idx, len(self.split_ratio) == 3)
        self._copy_files(li, class_dir)

    def _setup_files(self, class_dir):
        files = list_files(class_dir, IMAGE_EXTENSIONS)
        files.sort()
        random.shuffle(files)
        return files
    
    def _split_files(self, files, split_train_idx, split_val_idx, use_test, max_test=None):
        files_train = files[:split_train_idx]
        files_val = (
            files[split_train_idx:split_val_idx] if use_test else files[split_train_idx:]
        )

        li = [(files_train, "train"), (files_val, "val")]

        # optional test folder
        if use_test:
            files_test = files[split_val_idx:]
            if max_test is not None:
                files_test = files_test[:max_test]

            li.append((files_test, "test"))
        return li
    
    def _copy_files(self, files_type, class_dir, output='data'):

        # get the last part within the file
        class_name = path.split(class_dir)[1]
        for (files, folder_type) in files_type:
            full_path = path.join(output, folder_type, class_name)

            Path(full_path).mkdir(parents=True, exist_ok=True)
            for f in files:
                if type(f) is tuple:
                    for x in f:
                        copy2(str(x), str(full_path))
                else:
                    copy2(str(f), str(full_path))