from typing import List

class Data:
    def __init__(self) -> None:
        self.time: float
        self.iters: int
        self.dim_x: int
        self.dim_y: int
        self.point_data: int
        self.u: List[List[float]] = []
        self.v: List[List[float]] = []
        self.p: List[List[float]] = []
        self.flag: List[List[int]] = []

    def __str__(self) -> str:
        string = ""
        for attribute, value in self.__dict__.items():
            string += f"{attribute}: {value}\n"
        return string