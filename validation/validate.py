from typing import List

from data import Data

FILEPATH = "F:\\Documents\\Uni\\HPC\\Assessment\\vortex-shedding\\out\\vortex.vtk"
BENCHMARK_FILEPATH = "F:\\Documents\\Uni\\HPC\\Assessment\\vortex-shedding\\benchmarks\\main\\default.vtk"

class Validator:

    @staticmethod
    def validate() -> bool:
        benchmark = Validator.read_file(BENCHMARK_FILEPATH)
        result = Validator.read_file(FILEPATH)
        return benchmark == result

    @staticmethod
    def read_file(filepath: str) -> Data:
        data = Data()
        with open(filepath, 'r', encoding="utf-8") as file:
            text = file.read()
            tables = text.split("LOOKUP_TABLE default")
            data = Validator.parse_header(tables[0], data)
            data.u = Validator.parse_float_table(tables[1])
            data.v = Validator.parse_float_table(tables[2])
            data.p = Validator.parse_float_table(tables[3])
            data.flag = Validator.parse_int_table(tables[4])
        return data

    @staticmethod
    def parse_header(text: str, data: Data) -> Data:
        feilds = text.split("\n")
        data.time = float(feilds[6])
        data.iters = int(feilds[8])
        dims = feilds[9].split(" ")
        data.dim_x = int(dims[1])
        data.dim_y = int(dims[2])
        point_data = feilds[12].split(" ")
        data.point_data = int(point_data[1])
        return data

    @staticmethod
    def parse_float_table(text: str) -> List[List[float]]:
        storage = []
        lines = Validator.split_table(text)
        for line in lines:
            values = line.strip().split(" ")
            row = []
            for value in values:
                row.append(float(value))
            storage.append(row)
        return storage

    @staticmethod
    def parse_int_table(text: str) -> List[List[int]]:
        storage = []
        lines = Validator.split_table(text)
        for line in lines:
            values = line.strip().split(" ")
            row = []
            for value in values:
                row.append(int(value))
            storage.append(row)
        return storage

    @staticmethod
    def split_table(text: str) -> List[str]:
        lines = text.split("\n")[:-1]
        if "SCALARS" in lines[-1]:
            lines = lines[:-2]
        return lines[1:]

def main() -> None:
    assert Validator.validate()

if __name__ == "__main__":
    main()
