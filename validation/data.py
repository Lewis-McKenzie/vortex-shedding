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
    
    def u_diffs(self, other: object) -> List[float]:
        u_diffs = []
        for i, row in enumerate(self.u):
            for j, u in enumerate(row):
                if u != other.u[i][j]:
                    #print(f"u ({i}, {j}) diff: {abs(u - other.u[i][j])}")
                    u_diffs.append(abs(u - other.u[i][j]))
        return u_diffs

    def v_diffs(self, other: object) -> List[float]:
        v_diffs = []
        for i, row in enumerate(self.v):
            for j, v in enumerate(row):
                if v != other.v[i][j]:
                    #print(f"v ({i}, {j}) diff: {abs(v - other.v[i][j])}")
                    v_diffs.append(abs(v - other.v[i][j]))
        return v_diffs
    
    def p_diffs(self, other: object) -> List[float]:
        p_diffs = []
        for i, row in enumerate(self.p):
            for j, p in enumerate(row):
                if p != other.p[i][j]:
                    #print(f"p ({i}, {j}) diff: {abs(p - other.p[i][j])}")
                    p_diffs.append(abs(p - other.p[i][j]))
        return p_diffs

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Data):
            return False
        if (self.dim_x, self.dim_y) != (__o.dim_x, __o.dim_y):
            return False
        return self.u_diffs(__o) == [] and self.v_diffs(__o) == [] and self.p_diffs(__o) == []
