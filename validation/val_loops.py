from typing import List


def parse() -> List[List[int]]:
    with open("./log", 'r', encoding="utf-8") as file:
        txt = file.read()
        csv = txt.split("###")[1].split("\n")[1:-1]
        #print(csv)
        csv = [c.split(",") for c in csv]
        return [[int(i), int(i_lim), int(j), int(j_lim)] for i, i_lim, j, j_lim in csv]

def setup():
    return [[i+1, i+2, j+1, j+9] for i in range(512) for j in []]

def comp():
    data = parse()
    real = setup()
    print(real)






def main() -> None:
    comp()

if __name__ == "__main__":
    main()
