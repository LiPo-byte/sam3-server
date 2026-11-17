"""Collect torch/torchvision and their full dependency closure from an existing venv into a staging dir.

Run it with the source venv's python, e.g.:
    ~/venvs/sam2/bin/python collect_torch.py ./_site
Files are hard-linked when possible (no extra disk usage), copied otherwise.
"""
import os
import re
import shutil
import sys
from importlib.metadata import PackageNotFoundError, distribution, distributions

try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement

ROOTS = ["torch", "torchvision"]
# CUDA runtime wheels torch loads at import time; always staged even if torch's metadata
# reaches them only through extras of an intermediate package
ALWAYS_PREFIXES = ("nvidia-",)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def resolve_closure(roots):
    seen, done_extras, missing = {}, {}, []
    stack = [(r.name, frozenset(normalize(x) for x in r.extras)) for r in map(Requirement, roots)]
    stack += [(d.metadata["Name"], frozenset()) for d in distributions()
              if normalize(d.metadata["Name"]).startswith(ALWAYS_PREFIXES)]
    while stack:
        raw_name, extras = stack.pop()
        name = normalize(raw_name)
        if name not in seen:
            try:
                seen[name] = distribution(name)
            except PackageNotFoundError:
                missing.append(name)
                continue
            done_extras[name] = set()

        # base requirements once, plus each requested extra once
        pending = [e for e in [""] + sorted(extras) if e not in done_extras[name]]
        if not pending:
            continue
        done_extras[name].update(pending)

        for raw in seen[name].requires or []:
            req = Requirement(raw)
            if req.marker is None or any(req.marker.evaluate({"extra": e}) for e in pending):
                stack.append((req.name, frozenset(normalize(x) for x in req.extras)))
    return seen, sorted(set(missing))


def stage(dists, dest):
    os.makedirs(dest, exist_ok=True)
    linked = copied = 0
    for dist in dists.values():
        for f in dist.files or []:
            # entries outside site-packages (e.g. ../../../bin/torchrun) are not needed in the image
            if str(f).startswith(".."):
                continue
            src = f.locate()
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dest, str(f))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                continue
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                shutil.copy2(src, dst)
                copied += 1
    return linked, copied


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <staging_dir>")
    dest = sys.argv[1]
    if os.path.exists(dest):
        shutil.rmtree(dest)

    dists, missing = resolve_closure(ROOTS)
    print(f"python {sys.version.split()[0]}, {len(dists)} packages:")
    for name, dist in sorted(dists.items()):
        print(f"  {name}=={dist.version}")
    if missing:
        print(f"WARNING: not installed in source venv: {', '.join(sorted(missing))}")

    linked, copied = stage(dists, dest)
    print(f"staged into {dest}: {linked} hard-linked, {copied} copied")


if __name__ == "__main__":
    main()
