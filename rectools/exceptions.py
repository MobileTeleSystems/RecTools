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

"""Exceptions module"""


class NotFittedError(Exception):
    """The error is raised when some fittable object is attempted to be used without fitting first."""

    def __init__(self, obj_name: str) -> None:
        super().__init__()
        self.obj_name = obj_name

    def __str__(self) -> str:
        return f"{self.obj_name} isn't fitted, call method `fit` first."
