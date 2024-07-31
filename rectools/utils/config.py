from pydantic import BaseModel


class BaseConfig(BaseModel, extra="forbid"):
    """Base config class for rectools."""

    pass
