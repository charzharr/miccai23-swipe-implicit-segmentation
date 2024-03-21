""" 
Objects that create neural network components or modules so that architecture
definitions can be cleaner & more generalizable. 

e.g. BN -> Norm('batchnorm', **kwargs) in network definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormFactory3d:
    
    def __init__(self, identifier, groups=None):
        """
        Args:
            Size (int): either #channels or tensor shape for layer norm.
        """
        
        self._groups = groups
        
        if isinstance(identifier, str):
            name = identifier.lower()
            if 'batch' in name:
                self._name = 'BatchNorm3d'
                self.norm = nn.BatchNorm3d
            elif 'instance' in name:
                self._name = 'InstanceNorm3d'
                self.norm = nn.InstanceNorm3d
            # elif 'layer' in name:
            #     self.norm = nn.LayerNorm
            elif 'group' in name:
                from numbers import Number
                assert groups is not None and isinstance(groups, Number)
                self._name = 'GroupNorm3d'
                self.norm = nn.GroupNorm
            else:
                raise ValueError(f'Given norm name "{name}" is not supported.')
        elif identifier is NormFactory3d:
            self._name = identifier.name
            self.norm = identifier.norm
        elif isinstance(identifier, nn.Module):
            self.norm = identifier
        else:
            raise ValueError(f'Constructor must be given str, module, or fact.')
    
    def create(self, *args, **kwargs):
        if self.norm is nn.GroupNorm:
            return self.norm(self._groups, *args, **kwargs)
        return self.norm(*args, **kwargs)
    
    def adjust_channels_by_factor(self, channels, factor=None):
        if factor is None:
            if 'groupnorm' in self.name.lower():
                factor = self._groups
            else:
                return int(channels)
        rem = channels % factor
        if rem < factor / 2:
            adj_channels = channels - rem 
            # print(channels, '->', adj_channels)
            return int(adj_channels)
        adj_channels = channels + (factor - rem)
        # print(channels, '->', adj_channels)
        return int(adj_channels)
    
    def __repr__(self):
        string = f'(NormFactory3d) - {self.name}'
        if 'groupnorm' in self.name.lower():
            string += f'\n   groups={self._groups}'
        return string
        
    @property
    def name(self):
        return str(self._name)
    
    
class NormFactory2d:
    
    def __init__(self, identifier, groups=None):
        """
        Args:
            Size (int): either #channels or tensor shape for layer norm.
        """
        
        self._groups = groups
        
        if isinstance(identifier, str):
            name = identifier.lower()
            if 'batch' in name:
                self._name = 'BatchNorm2d'
                self.norm = nn.BatchNorm2d
            elif 'instance' in name:
                self._name = 'InstanceNorm2d'
                self.norm = nn.InstanceNorm2d
            # elif 'layer' in name:
            #     self.norm = nn.LayerNorm
            elif 'group' in name:
                from numbers import Number
                assert groups is not None and isinstance(groups, Number)
                self._name = 'GroupNorm2d'
                self.norm = nn.GroupNorm
            else:
                raise ValueError(f'Given norm name "{name}" is not supported.')
        elif identifier is NormFactory3d:
            self._name = identifier.name
            self.norm = identifier.norm
        elif isinstance(identifier, nn.Module):
            self.norm = identifier
        else:
            raise ValueError(f'Constructor must be given str, module, or fact.')
    
    def create(self, *args, **kwargs):
        if self.norm is nn.GroupNorm:
            return self.norm(self._groups, *args, **kwargs)
        return self.norm(*args, **kwargs)
    
    def adjust_channels_by_factor(self, channels, factor=None):
        if factor is None:
            if 'groupnorm' in self.name.lower():
                factor = self._groups
            else:
                return int(channels)
        rem = channels % factor
        if rem < factor / 2:
            adj_channels = channels - rem 
            # print(channels, '->', adj_channels)
            return int(adj_channels)
        adj_channels = channels + (factor - rem)
        # print(channels, '->', adj_channels)
        return int(adj_channels)
    
    def __repr__(self):
        string = f'(NormFactory2d) - {self.name}'
        if 'groupnorm' in self.name.lower():
            string += f'\n   groups={self._groups}'
        return string
        
    @property
    def name(self):
        return str(self._name)


class ActFactory:
    
    def __init__(self, identifier):
        if isinstance(identifier, str):
            name = identifier.lower()
            if name == 'relu':
                self._name = 'ReLU'
                self.act = nn.ReLU
            elif name == 'leakyrelu':
                self._name = 'LeakyReLU'
                self.act = nn.LeakyReLU
            elif name == 'prelu':
                self._name = 'PReLU'
                self.act = nn.PReLU
            elif name == 'sigmoid':
                self._name = 'Sigmoid'
                self.act = nn.Sigmoid
            elif name == 'elu':
                self._name = 'ELU'
                self.act = nn.ELU
            elif name == 'gelu':
                self._name = 'GELU'
                self.act = nn.GELU
            else:
                raise ValueError(f'Given act name "{name}" is not supported.')
        elif identifier is ActFactory:
            self._name = identifier.name
            self.act = identifier.act
        elif isinstance(identifier, nn.Module):
            self.act = identifier
        else:
            raise ValueError(f'Constructor must be given str, module, or fact.')

    def create(self, *args, **kwargs):
        return self.act(*args, **kwargs)
    
    @property 
    def name(self):
        return str(self._name)
    
    def __repr__(self):
        string = f'(ActFactory) - {self.name}'
        return string

