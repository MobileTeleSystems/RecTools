from pydantic import BaseModel


class BaseConfig(BaseModel, extra="forbid"):
    pass