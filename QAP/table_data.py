import re
from pathlib import Path
from QAP import core as base
ROOT = Path(__file__).resolve().parent

CLASSES = "bur chr esc had kra lipa nug rou scr sko ste tai tho wil".split()


def relative_gap(cost, reference):
    if reference == 0:
        if cost != 0:
            raise ValueError(f"Percentage gap is undefined for positive cost {cost} against a zero reference")
        return 0.0
    return 100*(cost/reference-1)


def qaplib_data():
    records, excluded = [], []
    for filename in base.list_qaplib_instances(str(ROOT/"qaplib")):
        path = Path(filename)
        name = path.name.split(".")[0]
        n, A, B = base.parse_qaplib_dat(filename)
        category = re.match(r"[a-zA-Z]+",name).group().lower()
        if not 12 <= n <= 64 or category not in CLASSES:
            excluded.append({"name":name,"n":n,"reason":"outside explicit 12..64 / 14-class table scope"})
            continue
        solution = base.find_solution_file(str(ROOT/"qapsoln"),name)
        known, permutation = base.parse_qaplib_sln(solution,n_expected=n)
        if permutation is None:
            excluded.append({"name":name,"n":n,"reason":"solution file has no valid permutation"})
            continue
        mode = base.detect_solution_mode(A[None],B[None],known,permutation[None])
        if mode is None:
            excluded.append({"name":name,"n":n,"reason":"supplied solution objective and permutation inconsistent"})
            continue
        records.append({"name":name,"n":n,"class":category,"A":A,"B":B,
                        "known":known,"swap_ab":mode.swap_ab,"use_inv":mode.use_inv})
    assert set(r["class"] for r in records) == set(CLASSES)
    return records, excluded
