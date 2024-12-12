#  Copyright 2022-2024 MTS (Mobile Telesystems)
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

import pickle
from collections.abc import Hashable
from typing import Callable, Dict, List, Union

import numpy as np
import pytest

from rectools.dataset import IdMap
from rectools.tools import ItemToItemAnnRecommender, UserToItemAnnRecommender


class TestItemToItemAnnRecommender:
    @pytest.fixture
    def recommender(self) -> Callable[[bool], ItemToItemAnnRecommender]:
        def make_recommender(with_id_map: bool) -> ItemToItemAnnRecommender:
            index_init_params = {"method": "hnsw", "space": "cosinesimil"}
            index_query_time_params = {"efSearch": 100}
            create_index_params = {"M": 100, "efConstruction": 100, "post": 0}
            recommender_conf = {"top_n": 2, "index_top_k": 3}
            item_vectors = np.array(
                [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
            )
            item_map: Dict[Hashable, int] = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
            item_id_to_item_vector_id: Union[IdMap, Dict[Hashable, int]] = (
                IdMap.from_dict(item_map) if with_id_map else item_map
            )

            rec = ItemToItemAnnRecommender(
                item_vectors=item_vectors,
                item_id_map=item_id_to_item_vector_id,
                index_top_k=recommender_conf["index_top_k"],
                index_init_params=index_init_params,
                index_query_time_params=index_query_time_params,
                create_index_params=create_index_params,
            )
            return rec.fit()

        return make_recommender

    @pytest.fixture
    def recommender_default(self) -> Callable[[bool], ItemToItemAnnRecommender]:
        def make_recommender(with_id_map: bool) -> ItemToItemAnnRecommender:
            index_init_params = {"method": "hnsw", "space": "cosinesimil"}
            index_query_time_params = {"efSearch": 100}
            create_index_params = {"M": 100, "efConstruction": 100, "post": 0}
            recommender_conf = {"top_n": 2, "index_top_k": 3}
            item_vectors = np.array(
                [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
            )
            item_map: Dict[Hashable, int] = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
            item_id_to_item_vector_id: Union[IdMap, Dict[Hashable, int]] = (
                IdMap.from_dict(item_map) if with_id_map else item_map
            )

            rec = ItemToItemAnnRecommender(
                item_vectors=item_vectors,
                item_id_map=item_id_to_item_vector_id,
                index_top_k=recommender_conf["index_top_k"],
                index_init_params=index_init_params,
                index_query_time_params=index_query_time_params,
                create_index_params=create_index_params,
            )
            return rec.fit()

        return make_recommender

    @pytest.mark.parametrize(
        "iid,expected,with_id_map",
        (
            ("0", ["1", "4"], False),
            ("0", ["1", "4"], True),
        ),
    )
    def test_get_item_list_for_item_default(
        self,
        recommender_default: Callable[[bool], ItemToItemAnnRecommender],
        iid: str,
        expected: List[str],
        with_id_map: bool,
    ) -> None:
        recommender_ = recommender_default(with_id_map)
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0])) if str(x) != iid]
        res = recommender_.get_item_list_for_item(item_id=iid, top_n=top_n, item_available_ids=item_ids)
        assert len(res) == top_n
        assert len(set(res).intersection(set(item_ids))) == top_n
        assert list(res) == expected

    @pytest.mark.parametrize(
        "iid,expected,with_id_map",
        (
            ("0", ["1", "4"], False),
            ("0", ["1", "4"], True),
        ),
    )
    def test_get_item_list_for_item(
        self, recommender: Callable[[bool], ItemToItemAnnRecommender], iid: str, expected: List[str], with_id_map: bool
    ) -> None:
        recommender_ = recommender(with_id_map)
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0])) if str(x) != iid]
        res = recommender_.get_item_list_for_item(item_id=iid, top_n=top_n, item_available_ids=item_ids)
        assert len(res) == top_n
        assert len(set(res).intersection(set(item_ids))) == top_n
        assert list(res) == expected

    @pytest.mark.parametrize("iid,expected,with_id_map", (("0", ["1", "4"], False), ("0", ["1", "4"], True)))
    def test_get_item_list_for_item_full(
        self, recommender: Callable[[int], ItemToItemAnnRecommender], iid: str, expected: List[str], with_id_map: bool
    ) -> None:
        recommender_ = recommender(with_id_map)
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0]))]
        res = recommender_.get_item_list_for_item(item_id=iid, top_n=top_n)
        assert len(res) == top_n
        assert len(set(res).intersection(set(item_ids))) == top_n
        assert list(res) == expected

    @pytest.mark.parametrize(
        "iid,expected,with_id_map",
        (
            (["0", "1"], [["4", "3"], ["4", "3"]], False),
            (["0", "1"], [["4", "3"], ["4", "3"]], True),
        ),
    )
    def test_get_item_list_for_item_batch(
        self,
        recommender: Callable[[int], ItemToItemAnnRecommender],
        iid: List[str],
        expected: List[List[str]],
        with_id_map: bool,
    ) -> None:
        recommender_ = recommender(with_id_map)
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0])) if str(x) not in iid]
        res = recommender_.get_item_list_for_item_batch(
            item_ids=iid, top_n=top_n, item_available_ids=[item_ids, item_ids]
        )
        for item_res in res:
            assert len(item_res) == top_n
            assert len(set(item_res).intersection(set(item_ids))) == top_n
        for case_res, case_exp in zip(res, expected):
            assert list(case_res) == case_exp

    @pytest.mark.parametrize(
        "iid,expected,with_id_map",
        (
            (["0", "1"], [["1", "4"], ["0", "4"]], False),
            (["0", "1"], [["1", "4"], ["0", "4"]], True),
        ),
    )
    def test_get_item_list_for_user_batch_full(
        self,
        recommender: Callable[[int], ItemToItemAnnRecommender],
        iid: List[str],
        expected: List[List[str]],
        with_id_map: bool,
    ) -> None:
        recommender_ = recommender(with_id_map)
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0]))]
        res = recommender_.get_item_list_for_item_batch(item_ids=iid, top_n=top_n)
        for item_res in res:
            assert len(item_res) == top_n
            assert len(set(item_res).intersection(set(item_ids))) == top_n
        for case_res, case_exp in zip(res, expected):
            assert list(case_res) == case_exp

    def test_pickle_unpickle(self, recommender: Callable[[int], ItemToItemAnnRecommender]) -> None:
        recommender_ = recommender(0)
        dump = pickle.dumps(recommender_)
        loaded_dump = pickle.loads(dump)
        iid = "0"
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0])) if str(x) != iid]
        res_origin = recommender_.get_item_list_for_item(item_id=iid, top_n=top_n, item_available_ids=item_ids)
        res_loaded = loaded_dump.get_item_list_for_item(item_id=iid, top_n=top_n, item_available_ids=item_ids)
        # indexes are always unequal, hence there is no point in asserting index equality
        assert np.allclose(recommender_.item_vectors, loaded_dump.item_vectors)
        assert list(recommender_.item_id_map.get_external_sorted_by_internal()) == list(
            loaded_dump.item_id_map.get_external_sorted_by_internal()
        )
        assert (res_origin == res_loaded).all()


class TestUserToItemAnnRecommender:
    @pytest.fixture
    def recommender(self) -> Callable[[bool], UserToItemAnnRecommender]:
        def make_recommender(with_id_map: bool) -> UserToItemAnnRecommender:
            index_init_params = {"method": "hnsw", "space": "cosinesimil"}
            index_query_time_params = {"efSearch": 100}
            create_index_params = {"M": 100, "efConstruction": 100, "post": 0}
            recommender_conf = {"top_n": 2, "index_top_k": 3}
            user_vectors = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
            item_vectors = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
            user_map: Dict[Hashable, int] = {"0": 0, "1": 1}
            item_map: Dict[Hashable, int] = {"0": 0, "1": 1, "2": 2}
            user_id_to_user_vector_id: Union[IdMap, Dict[Hashable, int]] = (
                IdMap.from_dict(user_map) if with_id_map else user_map
            )
            item_id_to_item_vector_id: Union[IdMap, Dict[Hashable, int]] = (
                IdMap.from_dict(item_map) if with_id_map else item_map
            )

            rec = UserToItemAnnRecommender(
                user_vectors=user_vectors,
                item_vectors=item_vectors,
                user_id_map=user_id_to_user_vector_id,
                item_id_map=item_id_to_item_vector_id,
                index_top_k=recommender_conf["index_top_k"],
                index_init_params=index_init_params,
                index_query_time_params=index_query_time_params,
                create_index_params=create_index_params,
            )
            return rec.fit()

        return make_recommender

    @pytest.fixture
    def recommender_default(self) -> Callable[[bool], UserToItemAnnRecommender]:
        def make_recommender(with_id_map: bool) -> UserToItemAnnRecommender:
            recommender_conf = {"top_n": 2, "index_top_k": 3}
            user_vectors = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
            item_vectors = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
            user_map: Dict[Hashable, int] = {"0": 0, "1": 1}
            item_map: Dict[Hashable, int] = {"0": 0, "1": 1, "2": 2}
            user_id_to_user_vector_id: Union[IdMap, Dict[Hashable, int]] = (
                IdMap.from_dict(user_map) if with_id_map else user_map
            )
            item_id_to_item_vector_id: Union[IdMap, Dict[Hashable, int]] = (
                IdMap.from_dict(item_map) if with_id_map else item_map
            )

            rec = UserToItemAnnRecommender(
                user_vectors=user_vectors,
                item_vectors=item_vectors,
                user_id_map=user_id_to_user_vector_id,
                item_id_map=item_id_to_item_vector_id,
                index_top_k=recommender_conf["index_top_k"],
            )
            return rec.fit()

        return make_recommender

    @pytest.fixture
    def recommender_broken(self) -> Callable[[bool], UserToItemAnnRecommender]:
        def make_recommender(with_id_map: bool) -> UserToItemAnnRecommender:
            recommender_conf = {"top_n": 2, "index_top_k": 3}
            user_vectors = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
            item_vectors = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
            user_map: Dict[Hashable, int] = {"0": 0, "1": 1}
            item_map: Dict[Hashable, int] = {"0": 0, "1": 1, "2": 2}
            user_id_to_user_vector_id: Union[IdMap, Dict[Hashable, int]] = (
                IdMap.from_dict(user_map) if with_id_map else user_map
            )
            item_id_to_item_vector_id: Union[IdMap, Dict[Hashable, int]] = (
                IdMap.from_dict(item_map) if with_id_map else item_map
            )

            rec = UserToItemAnnRecommender(
                user_vectors=user_vectors,
                item_vectors=item_vectors,
                user_id_map=user_id_to_user_vector_id,
                item_id_map=item_id_to_item_vector_id,
                index_top_k=recommender_conf["index_top_k"],
            )
            return rec.fit()

        return make_recommender

    @pytest.mark.parametrize(
        "with_id_map",
        ((False,), (True,)),
    )
    def test_raises_shape_mismatch(
        self, recommender_broken: Callable[[bool], UserToItemAnnRecommender], with_id_map: bool
    ) -> None:
        with pytest.raises(ValueError):
            recommender_broken(with_id_map)

    @pytest.mark.parametrize(
        "uid,expected,with_id_map",
        (
            ("0", ["2", "1"], False),
            ("0", ["2", "1"], False),
        ),
    )
    def test_get_item_list_for_user_default(
        self,
        recommender_default: Callable[[bool], UserToItemAnnRecommender],
        uid: str,
        expected: List[str],
        with_id_map: bool,
    ) -> None:
        recommender_ = recommender_default(with_id_map)
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0]))]
        res = recommender_.get_item_list_for_user(user_id=uid, top_n=top_n, item_ids=item_ids)
        assert len(res) == top_n
        assert len(set(res).intersection(set(item_ids))) == top_n
        assert list(res) == expected

    @pytest.mark.parametrize(
        "uid,expected,with_id_map",
        (
            ("0", ["2", "1"], False),
            ("0", ["2", "1"], True),
        ),
    )
    def test_get_item_list_for_user(
        self, recommender: Callable[[bool], UserToItemAnnRecommender], uid: str, expected: List[str], with_id_map: bool
    ) -> None:
        recommender_ = recommender(with_id_map)
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0]))]
        res = recommender_.get_item_list_for_user(user_id=uid, top_n=top_n, item_ids=item_ids)
        assert len(res) == top_n
        assert len(set(res).intersection(set(item_ids))) == top_n
        assert list(res) == expected

    @pytest.mark.parametrize(
        "uid,expected,with_id_map",
        (
            ("0", ["2", "1"], False),
            ("0", ["2", "1"], True),
        ),
    )
    def test_get_item_list_for_user_full(
        self, recommender: Callable[[bool], UserToItemAnnRecommender], uid: str, expected: List[str], with_id_map: bool
    ) -> None:
        recommender_ = recommender(with_id_map)
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0]))]
        res = recommender_.get_item_list_for_user(user_id=uid, top_n=top_n)
        assert len(res) == top_n
        assert len(set(res).intersection(set(item_ids))) == top_n
        assert list(res) == expected

    @pytest.mark.parametrize(
        "uid,expected,with_id_map",
        (
            (["0", "1"], [["2", "1"], ["2", "1"]], False),
            (["0", "1"], [["2", "1"], ["2", "1"]], True),
        ),
    )
    def test_get_item_list_for_user_batch(
        self,
        recommender: Callable[[bool], UserToItemAnnRecommender],
        uid: List[str],
        expected: List[List[str]],
        with_id_map: bool,
    ) -> None:
        recommender_ = recommender(with_id_map)
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0]))]
        res = recommender_.get_item_list_for_user_batch(user_ids=uid, top_n=top_n, item_ids=[item_ids, item_ids])
        for user_res in res:
            assert len(user_res) == top_n
            assert len(set(user_res).intersection(set(item_ids))) == top_n
        for case_res, case_exp in zip(res, expected):
            assert list(case_res) == case_exp

    @pytest.mark.parametrize(
        "uid,expected,with_id_map",
        (
            (["0", "1"], [["2", "1"], ["2", "1"]], False),
            (["0", "1"], [["2", "1"], ["2", "1"]], True),
        ),
    )
    def test_get_item_list_for_user_batch_full(
        self,
        recommender: Callable[[bool], UserToItemAnnRecommender],
        uid: List[str],
        expected: List[List[str]],
        with_id_map: bool,
    ) -> None:
        recommender_ = recommender(with_id_map)
        top_n = 2
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0]))]
        res = recommender_.get_item_list_for_user_batch(user_ids=uid, top_n=top_n)
        for user_res in res:
            assert len(user_res) == top_n
            assert len(set(user_res).intersection(set(item_ids))) == top_n
        for case_res, case_exp in zip(res, expected):
            assert list(case_res) == case_exp

    def test_pickle_unpickle(self, recommender: Callable[[int], UserToItemAnnRecommender]) -> None:
        recommender_ = recommender(0)
        dump = pickle.dumps(recommender_)
        loaded_dump = pickle.loads(dump)
        uid = "0"
        item_ids = [str(x) for x in list(range(recommender_.item_vectors.shape[0]))]
        res_origin = recommender_.get_item_list_for_user(user_id=uid, top_n=2, item_ids=item_ids)
        res_loaded = loaded_dump.get_item_list_for_user(user_id=uid, top_n=2, item_ids=item_ids)
        # indexes are always unequal, hence there is no point in asserting index equality
        assert np.allclose(recommender_.user_vectors, loaded_dump.user_vectors)
        assert np.allclose(recommender_.item_vectors, loaded_dump.item_vectors)
        assert list(recommender_.user_id_map.get_external_sorted_by_internal()) == list(
            loaded_dump.user_id_map.get_external_sorted_by_internal()
        )
        assert list(recommender_.item_id_map.get_external_sorted_by_internal()) == list(
            loaded_dump.item_id_map.get_external_sorted_by_internal()
        )
        assert (res_origin == res_loaded).all()
