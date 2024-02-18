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


class RequirementUnavailable:
    """Base class for dummy classes, which are returned if there are no dependencies required for the original class"""

    requirement: str = NotImplemented

    def __new__(cls, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Raise ImportError when an attempt to instantiate an unavailable model is made"""
        raise ImportError(
            f"Requirement `{cls.requirement}` is not satisfied. Run `pip install rectools[{cls.requirement}]` "
            f"to install extra requirements before accessing {cls.__name__}."
        )


class LightFMWrapperModel(RequirementUnavailable):
    """Dummy class, which is returned if there are no dependencies required for the model"""

    requirement = "lightfm"


class DSSMModel(RequirementUnavailable):
    """Dummy class, which is returned if there are no dependencies required for the model"""

    requirement = "torch"


class ItemToItemAnnRecommender(RequirementUnavailable):
    """Dummy class, which is returned if there are no dependencies required for the model"""

    requirement = "nmslib"


class UserToItemAnnRecommender(RequirementUnavailable):
    """Dummy class, which is returned if there are no dependencies required for the model"""

    requirement = "nmslib"


class VisualApp(RequirementUnavailable):
    """Dummy class, which is returned if there are no dependencies required for the model"""

    requirement = "ipywidgets"


class ItemToItemVisualApp(RequirementUnavailable):
    """Dummy class, which is returned if there are no dependencies required for the model"""

    requirement = "ipywidgets"
