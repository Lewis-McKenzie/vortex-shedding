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

    print_stats("u", benchmark.u_diffs(result))
    print_stats("v", benchmark.v_diffs(result))
    print_stats("p", benchmark.p_diffs(result))

    assert valid, "Test result does not match the benchmark"

def print_stats(name: str, diffs) -> None:
    d_abs = sorted(diffs, key=lambda d: d[0][0])
    d_rel = sorted(diffs, key=lambda d: d[0][1])
    print(f"{name} diffs: {len(diffs)}\n"
          f"max abs err: {d_abs[-1][0][0]} {d_abs[-1][1]} min abs err: {d_abs[0][0][0]} {d_abs[0][1]}\n"
          f"max rel err: {d_rel[-1][0][1]}% {d_rel[-1][1]} min rel err: {d_rel[0][0][1]}% {d_rel[0][1]}\n"
          f"mean abs err: {sum(d[0][0] for d in diffs)/len(diffs)} median abs err: {d_abs[len(diffs)//2][0][0]}\n"
          f"mean rel err: {sum(d[0][1] for d in diffs)/len(diffs)}% median rel err: {d_rel[len(diffs)//2][0][1]}%\n")


def args() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(description="Validate a test vtk file against a benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("bench", help="Location of benchmark vtk file")
    parser.add_argument("test", help="Location of test vtk file")
    args = parser.parse_args()
    return args.bench, args.test

if __name__ == "__main__":
    main()
