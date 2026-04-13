from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


# Pre-Process modules
class BaseProcess(ABC):
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# Dataset
class BaseData(Dataset, ABC):
    def __init__(
        self,
        data_root: str,
        modalities: Optional[List[str]] = None,
        process: Optional[Dict[str, BaseProcess]] = None,
        **kwargs
    ):
        """Args:
            data_root:数据集的根目录
            modalities: 模态列表（应该和process方法一一对应）
            process:预处理方法，inherit BaseProcess class e.g. ({'video': VideoProcess(), 'imu': IMUProcess()})
        """
        self.data_root = Path(data_root)
        self.modalities = modalities
        self.process = process
        self.samples = self.load_samples()
        if len(self.samples) == 0:
            raise ValueError(f"No samples found under {self.data_root}.")
        print(f"Using modalities: {self.modalities}")

    @abstractmethod
    def load_samples(self) -> List[Dict]:
        """
        加载样本元数据列表（所有的sample）

        Returns:
            List[Dict]: 每个dict包含该样本的所有路径和标签信息
                       {
                           'id': str,
                           'label': int,
                           'video_path': Path,      # ?
                           'imu_path': Path,      # ?
                           ...
                       }
        """
        pass

    @abstractmethod
    def load_data(self, sample_info: Dict) -> Dict:
        """
        加载某一个数据点
        Args:
            sample_info: 包含路径和元数据的字典

        Returns:
            Dict: 加载的实际数据
                 {
                     'video': ndarray,
                     'imu':ndarray,
                     'label': int,
                     'id': str,
                 }
        """
        pass

    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        data = self.load_data(sample_info)
        if self.process:
            processed_data = {}
            for key, value in data.items():
                if key not in self.process:
                    raise KeyError(
                        f"No preprocessing method found for modality '{key}'. "
                        f"Available processors: {list(self.process.keys())}. "
                        f"Please provide a processor for '{key}' or remove it from the data."
                    )

                if self.process[key] is None:
                    raise ValueError(
                        f"Processor for modality '{key}' is None. "
                        f"Please provide a valid processor or use an identity processor."
                    )

                try:
                    processed_data[key] = self.process[key](value)
                except Exception as e:
                    raise RuntimeError(
                        f"Error processing modality '{key}' with {self.process[key]}: {str(e)}"
                    ) from e

            data = processed_data

        return data


# TODO: Add more pre-process here!
class Identity(BaseProcess):

    def __call__(self, data: Any) -> Any:
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize(BaseProcess):

    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ToTensor(BaseProcess):

    def __call__(self, data: np.ndarray) -> Any:
        import torch
        return torch.from_numpy(data).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class VideoProcess(BaseProcess):
    def __init__(self, num_frames: int) -> None:
        self.num_frames = num_frames

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        # TODO
        T = len(frames)
        if T <= self.num_frames:
            indices = list(range(T)) + [T - 1] * (self.num_frames - T)
        indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
        return frames[indices]

    def __repr__(self) -> str:
        # TODO
        return f"{self.__class__.__name__}(num_frames={self.num_frames})'"


class IMUProcess(BaseProcess):
    def __init__(self, init=int) -> None:
        self.init = init

    def __call__(self, data: np.ndarray) -> np.ndarray:
        # TODO
        return super().__call__(data)

    def __repr__(self) -> str:
        # TODO
        return super().__repr__()


class Compose(BaseProcess):
    def __init__(self, processes: List[BaseProcess]):
        self.processes = processes

    def __call__(self, data: Any) -> Any:
        for process in self.processes:
            data = process(data)
        return data

    def __repr__(self) -> str:
        process_strs = [str(p) for p in self.processes]
        return f"{self.__class__.__name__}([\n  " + ",\n  ".join(process_strs) + "\n])"
