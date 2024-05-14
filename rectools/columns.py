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

"""Column names."""


class Columns:
    """Fixed column names for tables that contain interactions and recommendations."""

    User = "user_id"
    Item = "item_id"
    TargetItem = "target_item_id"
    Weight = "weight"
    Datetime = "datetime"
    Rank = "rank"
    Score = "score"
    Model = "model"
    UserItem = [User, Item]
    Interactions = [User, Item, Weight, Datetime]
    Recommendations = [User, Item, Score, Rank]
    RecommendationsI2I = [TargetItem, Item, Score, Rank]
