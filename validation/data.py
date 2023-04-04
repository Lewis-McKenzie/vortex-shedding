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

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Data):
            return False
        if (self.dim_x, self.dim_y) != (__o.dim_x, __o.dim_y):
            return False
        for i, row in enumerate(self.u):
            for j, u in enumerate(row):
                if u != __o.u[i][j]:
                    print(f"u ({i}, {j}) diff: {abs(u - __o.u[i][j])}")
                    return False
        for i, row in enumerate(self.v):
            for j, v in enumerate(row):
                if v != __o.v[i][j]:
                    print(f"v ({i}, {j}) diff: {abs(v - __o.v[i][j])}")
                    return False
        for i, row in enumerate(self.p):
            for j, p in enumerate(row):
                if p != __o.p[i][j]:
                    print(f"p ({i}, {j}) diff: {abs(p - __o.p[i][j])}")
                    return False
        return True
