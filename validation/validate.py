from typing import List

from data import Data

FILEPATH = "F:\\Documents\\Uni\\HPC\\Assessment\\vortex-shedding\\out\\vortex.vtk"

class Validator:
    data = Data()

    @staticmethod
    def read_file(filepath: str) -> None:
        with open(filepath, 'r', encoding="utf-8") as file:
            text = file.read()
            tables = text.split("LOOKUP_TABLE default")
            Validator.parse_header(tables[0])
            Validator.data.u = Validator.parse_float_table(tables[1])
            Validator.data.v = Validator.parse_float_table(tables[2])
            Validator.data.p = Validator.parse_float_table(tables[3])
            Validator.data.flag = Validator.parse_int_table(tables[4])
            d = Validator.data
            print()

    @staticmethod
    def parse_header(text: str) -> None:
        feilds = text.split("\n")
        print(feilds)
        Validator.data.time = float(feilds[6])
        Validator.data.iters = int(feilds[8])
        dims = feilds[9].split(" ")
        Validator.data.dim_x = int(dims[1])
        Validator.data.dim_x = int(dims[2])
        point_data = feilds[12].split(" ")
        Validator.data.point_data = int(point_data[1])

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
    Validator.read_file(FILEPATH)

if __name__ == "__main__":
    main()
