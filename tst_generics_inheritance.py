from pydantic import BaseModel
import typing as tp
import typing_extensions as tpe

class Conf1(BaseModel):
    x: int = 1


Conf_T = tp.TypeVar("Conf_T", bound=Conf1)

class Model1(tp.Generic[Conf_T]):
    def __init__(self, x: int = 10, y: int = 20, z: int = 30) -> None:
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_config(cls, config: Conf_T) -> tpe.Self:
        return cls(x=config.x)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})"
    
conf1 = Conf1()
model1 = Model1.from_config(conf1)
print(model1)

model1 = Model1()
print(model1)

class Conf2(Conf1):
    y: int = 2


Conf2_T = tp.TypeVar("Conf2_T", bound=Conf2)

class Model2(Model1):
    @classmethod
    def handle_x(cls, x: int) -> int:
        return x * 3
    
    @classmethod
    def from_config(cls, config: Conf2_T) -> tpe.Self:
        return cls(x=cls.handle_x(config.x), y=config.y)
    

conf2 = Conf2()
model2 = Model2.from_config(conf2)
print(model2)


model2 = Model2()
print(model2)


class Conf3(Conf2):
    z: int = 3

Conf3_T = tp.TypeVar("Conf3_T", bound=Conf3)

class Model3(Model2[Conf3_T]):
    @classmethod
    def from_config(cls, config: Conf3_T) -> tpe.Self:
        return cls(x=cls.handle_x(config.x), y=config.y, z=config.z)
    

conf3 = Conf3()
model3 = Model3.from_config(conf3)
print(model3)