from typing import List, Tuple
import argparse
 
from data import Data

class Validator:

    @staticmethod
    def validate(benchmark: Data, result: Data) -> bool:

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
    bench, test = args()
    benchmark = Validator.read_file(bench)
    result = Validator.read_file(test)
    valid = Validator.validate(benchmark, result)

    us = sorted(benchmark.u_diffs(result))
    vs = sorted(benchmark.v_diffs(result))
    ps = sorted(benchmark.p_diffs(result))
    print(f"u diffs: {len(us)} max: {max(us)} min: {min(us)} avg: {sum(us)/len(us)} medium: {us[len(us)//2]}")
    print(f"v diffs: {len(vs)} max: {max(vs)} min: {min(vs)} avg: {sum(vs)/len(vs)} medium: {us[len(vs)//2]}")
    print(f"p diffs: {len(ps)} max: {max(ps)} min: {min(ps)} avg: {sum(ps)/len(ps)} medium: {us[len(ps)//2]}")
    assert valid, "Test result does not match the benchmark"

def args() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(description="Validate a test vtk file against a benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("bench", help="Location of benchmark vtk file")
    parser.add_argument("test", help="Location of test vtk file")
    args = parser.parse_args()
    return args.bench, args.test

if __name__ == "__main__":
    main()
