from typing import List, Set


def parse() -> List[List[int]]:
    with open("./log", 'r', encoding="utf-8") as file:
        txt = file.read()
        csv = txt.split("###")[1].split("\n")[1:-1]
        #print(csv)
        csv = [c.split(",") for c in csv]
        return [f"{i} {i_lim} {j} {j_lim}" for i, i_lim, j, j_lim in csv]

def setup():
    l = []
    for i in range(512):
        for j in range(8):
            l.append(f"{i+1} {i+2} {j*16+1} {(j+1)*16+1}")
    return l

def comp():
    data = parse()
    real = setup()
    s1 = setify(data)
    s2 = setify(real)
    is_eq(s2, s1)

def setify(d) -> Set[List[int]]:
    s = set()
    for entry in d:
        if entry in s:
            print(f"dupe {entry}")
        s.add(entry)
    return s


def is_eq(real, test):
    for entry in real:
        if entry not in test:
            print(f"{entry} not in test")
    for entry in test:
        if entry not in real:
            print(f"{entry} not in real")






def main() -> None:
    comp()

if __name__ == "__main__":
    main()
