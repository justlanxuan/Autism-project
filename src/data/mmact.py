# mmact.py
from typing import Dict, List

from base import *

class MMActDataset(BaseData):
    def __init__(
            self,
            data_root:str,
            modalities: Optional[List[str]] = ['imu', 'video'],
            process: Optional[Dict[str, BaseProcess]] = None,
            **kwargs
    ):
        """
        MMAct dataset (2 modalities)
        """
        super().__init__(data_root, modalities, process, **kwargs)
    def load_samples(self) -> List[Dict]:
        # TODO
        return super().load_samples()
    def load_data(self, sample_info: Dict) -> Dict:
        # TODO
        return super().load_data(sample_info)

# How to use this?
dataset = MMActDataset(
    data_root = '/path',
    process = {
        'video': VideoProcess(num_frames=16),
        'imu': IMUProcess(init=0),
        'id': Identity(),
        "label": Identity()
    }
)
