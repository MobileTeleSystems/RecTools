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

import typing as tp


class NNModelUnavailable:
    """Dummy class the instance of which is returned in case a model provided lacks any libraries required"""

    def __new__(cls, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Raise ImportError when an attempt to instantiate an unavailable model is made"""
        raise ImportError(
            f"Cannot initialize {cls.__name__}: "
            f"run `pip install rectools[nn]` to install extra requirements before accessing {cls.__name__} "
            f"(see `extras/requirements-nn.txt)"
        )


class DSSMModel(NNModelUnavailable):
    """Dummy class the instance of which is returned in case DSSMModel lacks any libraries required"""
