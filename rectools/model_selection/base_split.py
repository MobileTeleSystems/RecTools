#  Copyright 2022 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""BaseSplitter."""

from abc import ABC, abstractmethod
from pprint import pprint


class BaseSplitter(ABC):
    """
    Base class for all Splitters.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def split(self):
        pass

    @abstractmethod
    def get_n_splits(self):
        pass
    
    def __repr__(self):
        return _build_repr(self)


def _build_repr(self):
    # This is copied from scikit-learn's BaseEstimator get_params
    cls = self.__class__
    init = cls.__init__
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted(
            [
                p.name
                for p in init_signature.parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
        )
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        value = getattr(self, key, None)
        params[key] = value
    
    params_str = ', '.join(f'{p}={v}' for p, v in params.items())
    return f'{class_name}({params_str})'

