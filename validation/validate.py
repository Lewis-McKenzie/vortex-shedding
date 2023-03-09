
FILEPATH = "/home/userfs/l/lm1797/Assessment/vortex-shedding/out/vortex.vtk"

class validator:
    @staticmethod
    def read_file(filepath: str) -> None:
        with open(filepath, 'r') as file:
            text = file.read().replace("\n", " ")
            tables = text.split("LOOKUP_TABLE default")
            print(tables[0])
            for table in tables[1:]:
                print(table)



def main() -> None:
    validator.read_file(FILEPATH)

if __name__ == "__main__":
    main()
