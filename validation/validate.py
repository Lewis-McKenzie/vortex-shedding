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
    if valid:
        print("Valid")
        return
    us = sorted(benchmark.u_diffs(result), key=lambda d: d[0])
    vs = sorted(benchmark.v_diffs(result), key=lambda d: d[0])
    ps = sorted(benchmark.p_diffs(result), key=lambda d: d[0])

    print(f"u diffs: {len(us)} max: {us[-1]} min: {us[0][0]} avg: {sum(d for d, _ in us)/len(us)} medium: {us[len(us)//2][0]}")
    print(f"v diffs: {len(vs)} max: {vs[-1]} min: {vs[0][0]} avg: {sum(d for d, _ in vs)/len(vs)} medium: {vs[len(vs)//2][0]}")
    print(f"p diffs: {len(ps)} max: {ps[-1]} min: {ps[0][0]} avg: {sum(d for d, _ in ps)/len(ps)} medium: {ps[len(ps)//2][0]}")
    assert valid, "Test result does not match the benchmark"

def args() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(description="Validate a test vtk file against a benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("bench", help="Location of benchmark vtk file")
    parser.add_argument("test", help="Location of test vtk file")
    args = parser.parse_args()
    return args.bench, args.test

if __name__ == "__main__":
    main()
