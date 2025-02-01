from rectools.utils.misc import unflatten_dict


class TestUnflattenDict:
    def test_empty(self) -> None:
        assert unflatten_dict({}) == {}

    def test_complex(self) -> None:
        flattened = {
            "a.b": 1,
            "a.c": 2,
            "d": 3,
            "a.e.f": [10, 20],
        }
        excepted = {
            "a": {"b": 1, "c": 2, "e": {"f": [10, 20]}},
            "d": 3,
        }
        assert unflatten_dict(flattened) == excepted

    def test_simple(self) -> None:
        flattened = {
            "a": 1,
            "b": 2,
        }
        excepted = {
            "a": 1,
            "b": 2,
        }
        assert unflatten_dict(flattened) == excepted

    def test_non_default_sep(self) -> None:
        flattened = {
            "a_b": 1,
            "a_c": 2,
            "d": 3,
        }
        excepted = {
            "a": {"b": 1, "c": 2},
            "d": 3,
        }
        assert unflatten_dict(flattened, sep="_") == excepted
