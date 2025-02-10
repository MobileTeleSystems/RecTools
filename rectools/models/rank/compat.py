from rectools.compat import RequirementUnavailable


class TorchRanker(RequirementUnavailable):
    """Dummy class, which is returned if there are no dependencies required for the model"""

    requirement = "torch"
