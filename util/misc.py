from torch import Tensor
from typing import Optional, List

class NestedTensor(object):
    def __init__(self, tensor : Tensor, mask : Optional[Tensor]):
        self.tensor = tensor
        if mask is None:
            self.mask = None
        else:
            self.mask = mask
    
    def to(self, device):
        cast_tensor = self.tensor.to(device)
        if self.mask is not None:
            cast_mask = self.mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)
    
    def decompose(self):
        return self.tensor, self.mask
        
    