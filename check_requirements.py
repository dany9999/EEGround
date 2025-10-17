
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple, Iterable, Set

try:
    # Python 3.8+
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    from packaging.requirements import Requirement
except Exception:
    print("Manca il modulo 'packaging'. Installa con: python3 -m pip install --user packaging",
          file=sys.stderr)
    sys.exit(2)

RESET = "\x1b[0m"; BOLD = "\x1b[1m"; GREEN = "\x1b[32m"; YELLOW = "\x1b[33m"; RED = "\x1b[31m"

def color(txt: str, c: str) -> str:
    return (c + txt + RESET) if sys.stdout.isatty() else txt

def parse_requirements_file(path: Path, seen: Set[Path] | None = None) -> list[str]:
    """Ritorna una lista "flat" di righe requirement, risolvendo eventuali '-r other.txt'."""
    if seen is None:
        seen = set()
    reqs: list[str] = []
    path = path.resolve()
    if path in seen:
        return reqs
    if not path.exists():
        print(color(f"[WARN] requirements non trovato: {path}", YELLOW), file=sys.stderr)
        return reqs
    seen.add(path)
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--"):  # opzioni pip globali
            continue
        if line.startswith("-e "):  # editable
            reqs.append(line)
            continue
        if line.startswith("-r ") or line.startswith("--requirement "):
            inc = line.split(maxsplit=1)[1].strip()
            reqs.extend(parse_requirements_file((path.parent / inc).resolve(), seen))
            continue
        reqs.append(line)
    return reqs

def check_req(line: str) -> Tuple[str, str, str, str]:
    """
    Ritorna (name, required_spec, installed_version|'-', status)
    status ∈ {OK, MISMATCH, MISSING, EDITABLE}
    """
    s = line.strip()
    if s.startswith("-e "):
        return (s, "-", "-", "EDITABLE")
    try:
        req = Requirement(s)
    except Exception:
        # Non PEP 508 (es. URL VCS) → segna EDITABLE/UNKNOWN
        return (s, "-", "-", "EDITABLE")
    name = req.name
    spec = str(req.specifier) if req.specifier else ""
    try:
        inst_ver = version(name)
    except PackageNotFoundError:
        return (name, spec or "-", "-", "MISSING")
    if not spec:  # nessun vincolo -> qualsiasi versione OK
        return (name, "-", inst_ver, "OK")
    ok = req.specifier.contains(inst_ver, prereleases=True)
    return (name, spec, inst_ver, "OK" if ok else "MISMATCH")

def print_table(rows: Iterable[Tuple[str, str, str, str]]) -> int:
    rows = list(rows)
    headers = ("package", "required", "installed", "status")
    data = [headers] + rows
    widths = [max(len(str(r[i])) for r in data) for i in range(4)]

    def fmt(r): return "  ".join(str(r[i]).ljust(widths[i]) for i in range(4))

    print(BOLD + fmt(headers) + RESET)
    bad = 0
    for name, spec, inst, status in rows:
        if status == "OK":
            st = color(status, GREEN)
        elif status == "MISSING":
            st = color(status, RED); bad += 1
        elif status == "MISMATCH":
            st = color(status, YELLOW); bad += 1
        else:
            st = color(status, YELLOW)
        print(f"{name.ljust(widths[0])}  {spec.ljust(widths[1])}  {inst.ljust(widths[2])}  {st}")
    return 1 if bad else 0

def show_torch_info() -> None:
    try:
        import torch
        print("\n" + BOLD + "PyTorch info" + RESET)
        print(f"  torch: {torch.__version__} | CUDA runtime: {getattr(torch.version,'cuda',None)} | available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"  CUDA device count: {torch.cuda.device_count()} | device0: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
    except Exception:
        pass

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Uso: python3 check_requirements.py /path/to/requirements.txt", file=sys.stderr)
        return 2
    req_file = Path(argv[1])
    req_lines = parse_requirements_file(req_file)
    results = [check_req(line) for line in req_lines]
    rc = print_table(results)
    show_torch_info()
    return rc

if __name__ == "__main__":
    sys.exit(main(sys.argv))
