from typing import List, Tuple

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

    def abs_err(self, a: float, b: float) -> float:
        return abs(a - b)

    def rel_err(self, a: float, b: float) -> float:
        return abs((a - b) / a) * 100
    
    def u_diffs(self, other: object) -> List[Tuple[Tuple[float, float], Tuple[int, int]]]:
        u_diffs = []
        for i, row in enumerate(self.u):
            for j, u in enumerate(row):
                if u != other.u[i][j]:
                    abs_err = self.abs_err(u, other.u[i][j])
                    rel_err = self.rel_err(u, other.u[i][j])
                    u_diffs.append(((abs_err, rel_err), (i, j)))
        return u_diffs

    def v_diffs(self, other: object) -> List[Tuple[Tuple[float, float], Tuple[int, int]]]:
        v_diffs = []
        for i, row in enumerate(self.v):
            for j, v in enumerate(row):
                if v != other.v[i][j]:
                    abs_err = self.abs_err(v, other.v[i][j])
                    rel_err = self.rel_err(v, other.v[i][j])
                    v_diffs.append(((abs_err, rel_err), (i, j)))
        return v_diffs
    
    def p_diffs(self, other: object) -> List[Tuple[Tuple[float, float], Tuple[int, int]]]:
        p_diffs = []
        for i, row in enumerate(self.p):
            for j, p in enumerate(row):
                if p != other.p[i][j]:
                    abs_err = self.abs_err(p, other.p[i][j])
                    rel_err = self.rel_err(p, other.p[i][j])
                    p_diffs.append(((abs_err, rel_err), (i, j)))
        return p_diffs

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Data):
            return False
        if (self.dim_x, self.dim_y) != (__o.dim_x, __o.dim_y):
            return False
        return self.u_diffs(__o) == [] and self.v_diffs(__o) == [] and self.p_diffs(__o) == []
