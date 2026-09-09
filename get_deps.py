import argparse
import contextlib
import keyword
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = REPO_ROOT / "third_party"

_FULL_SHA = re.compile(r"[0-9a-f]{40}")

_LEFTOVER = re.compile(r"^\.(?P<name>.+?)\.(?:tmp|stale)\.(?P<pid>\d+)(?:\.|$)")

_SERVER_REFUSES_SHA = re.compile(
    r"unadvertised object|allow-?tip-?sha1|allowanysha1inwant"
    r"|does not support shallow",
    re.IGNORECASE,
)

_REPO_SCOPED_GIT_VARS = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_INDEX_VERSION",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
        "GIT_CEILING_DIRECTORIES",
    }
)
_GIT_ENV = {k: v for k, v in os.environ.items() if k not in _REPO_SCOPED_GIT_VARS}


@dataclass(frozen=True)
class VendoredRepo:
    url: str
    commit: str
    comment: str

    @property
    def name(self) -> str:
        return module_name(self.url)


REPOS: tuple[VendoredRepo, ...] = (
    VendoredRepo(
        url="https://github.com/thuml/Time-Series-Library.git",
        commit="4e938a1767106324dd753b2a44832bf870a0252e",
        comment="TSLib: baseline forecasting models, layers, data loaders.",
    ),
    VendoredRepo(
        url="https://github.com/thuml/TimeXer.git",
        commit="76011909357972bd55a27adba2e1be994d81b327",
        comment=(
            "TimeXer: vendored for dataset/EPF, the five day-ahead "
            "electricity-price CSVs. The pinned TSLib dataset snapshot on "
            "Hugging Face has no EPF folder, so this tree is where the exact "
            "files behind the published Table 2 numbers actually live. The "
            "whole repository arrives — there is no sparse checkout here — so "
            "its near-fork of TSLib's models/ and layers/ is on disk too. "
            "Nothing imports it: the two bridge modules that put a vendored "
            "tree on sys.path name the Time-Series-Library and GCGNet."
        ),
    ),
    VendoredRepo(
        url="https://github.com/decisionintelligence/GCGNet.git",
        commit="4c48c87402ea8a975c14889c6aaf74ecebff222a",
        comment=(
            "GCGNet: the ICLR 2026 covariate-forecasting model, run as its "
            "authors wrote it. The repository is a fork of the TFB benchmark "
            "harness, so most of what arrives is a runner, a reporting app and "
            "thirty-odd other baselines that nothing here touches. Only the "
            "model and the layers it imports are used, through the `gcgnet` "
            "bridge module -- but the model reaches its layers by absolute "
            "path through the benchmark package, so the tree goes on sys.path "
            "whole rather than being loaded file by file."
        ),
    ),
    VendoredRepo(
        url="https://github.com/mumiao2000/CrossLinear.git",
        commit="d22366e2f59ced560a02b2b1c7cc673e3c02a13f",
        comment=(
            "CrossLinear: the KDD 2025 patch-linear forecaster, run as its "
            "authors wrote it. The repository is a fork of the "
            "Time-Series-Library, so it carries the same bare top-level "
            "models/, layers/ and data_provider/ packages -- which is why "
            "nothing puts this tree's own root on sys.path. The `crosslinear` "
            "bridge module puts third_party/ on the path instead and imports "
            "the model as crosslinear.models.CrossLinear, so the two forks "
            "cannot shadow each other. The model file needs nothing but torch "
            "and math, so that namespaced import is all it takes."
        ),
    ),
)


class VendorError(Exception):
    pass


def module_name(url: str) -> str:
    segment = url.rstrip("/").rsplit("/", 1)[-1]
    segment = segment.removesuffix(".git")
    name = re.sub(r"[^0-9a-z]+", "_", segment.lower()).strip("_")
    if name[:1].isdigit():
        name = "_" + name
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{url}: cannot derive a legal module name (got {name!r})")
    return name


def validate_manifest(repos: tuple[VendoredRepo, ...]) -> None:
    claimed: dict[str, str] = {}
    for repo in repos:
        if not _FULL_SHA.fullmatch(repo.commit):
            raise ValueError(
                f"{repo.url}: commit must be a full 40-character SHA, "
                f"got {repo.commit!r}"
            )
        name = repo.name
        if name in claimed:
            raise ValueError(
                f"{repo.url} and {claimed[name]} would both vendor as "
                f"third_party/{name}"
            )
        claimed[name] = repo.url


def _git(
    args: list[str], cwd: Path, *, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=_GIT_ENV,
            text=True,
            capture_output=capture,
            check=False,
        )
    except OSError as exc:
        raise VendorError(f"could not run git: {exc}") from None
    if check and proc.returncode != 0:
        raise VendorError(f"git {' '.join(args)} failed: {_last_line(proc)}")
    return proc


def _last_line(proc: subprocess.CompletedProcess[str]) -> str:
    lines = ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()
    return lines[-1] if lines else f"exit status {proc.returncode}"


def _exists(path: Path) -> bool:
    return path.is_symlink() or path.exists()


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _default_dir_mode() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return 0o777 & ~umask


def head_commit(checkout: Path) -> str | None:
    if not (checkout / ".git").exists():
        return None
    proc = _git(["rev-parse", "HEAD"], cwd=checkout, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def is_dirty(checkout: Path) -> bool:
    proc = _git(["status", "--porcelain"], cwd=checkout, check=False)
    return proc.returncode != 0 or bool(proc.stdout.strip())


def status(repo: VendoredRepo) -> str:
    dest = VENDOR_DIR / repo.name
    if not _exists(dest):
        return "missing"
    head = head_commit(dest)
    if head is None:
        return "not a git checkout"
    dirty = is_dirty(dest)
    if head != repo.commit:
        moved = f"pinned SHA changed (on {head[:12]})"
        return f"locally modified, and {moved}" if dirty else moved
    return "locally modified" if dirty else "up to date"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _sweep_orphans(name: str) -> int:
    if not VENDOR_DIR.is_dir():
        return 0
    removed = 0
    for entry in VENDOR_DIR.iterdir():
        match = _LEFTOVER.match(entry.name)
        if match and match["name"] == name and not _pid_alive(int(match["pid"])):
            _remove(entry)
            removed += 1
    return removed


def _install(src: Path, dest: Path) -> None:
    stale = dest.with_name(f".{dest.name}.stale.{os.getpid()}")
    _remove(stale)
    replaced = False
    try:
        if _exists(dest):
            os.replace(dest, stale)
            replaced = True
        os.replace(src, dest)
    except OSError as exc:
        if replaced and not _exists(dest):
            try:
                os.replace(stale, dest)
            except OSError:
                raise VendorError(
                    f"could not install {dest}: {exc}. The previous checkout is "
                    f"at {stale}; move it back by hand."
                ) from None
        raise VendorError(f"could not install {dest}: {exc}") from None
    if replaced:
        _remove(stale)


def vendor(repo: VendoredRepo) -> Path:
    dest = VENDOR_DIR / repo.name
    try:
        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        reclaimed = _sweep_orphans(repo.name)
        if reclaimed:
            plural = "y" if reclaimed == 1 else "ies"
            print(f"  reclaimed {reclaimed} leftover director{plural}")
        tmp = Path(
            tempfile.mkdtemp(prefix=f".{repo.name}.tmp.{os.getpid()}.", dir=VENDOR_DIR)
        )
    except OSError as exc:
        raise VendorError(f"could not prepare {VENDOR_DIR}: {exc}") from None

    installed = False
    try:
        _git(["init", "-q"], cwd=tmp)
        _git(["remote", "add", "origin", repo.url], cwd=tmp)
        shallow = _git(
            ["fetch", "-q", "--depth", "1", "origin", repo.commit],
            cwd=tmp,
            check=False,
        )
        if shallow.returncode == 0:
            _git(["checkout", "-q", "FETCH_HEAD"], cwd=tmp)
        elif _SERVER_REFUSES_SHA.search(shallow.stderr or ""):
            print(f"  {repo.name}: host refuses a bare SHA, fetching full history")
            _git(["fetch", "origin"], cwd=tmp, capture=False)
            _git(["checkout", "-q", repo.commit], cwd=tmp)
        else:
            raise VendorError(
                f"{repo.url}: could not fetch {repo.commit}: {_last_line(shallow)}"
            )
        head = head_commit(tmp)
        if head != repo.commit:
            raise VendorError(f"{repo.url}: checked out {head}, expected {repo.commit}")
        with contextlib.suppress(OSError):
            os.chmod(tmp, _default_dir_mode())
        _install(tmp, dest)
        installed = True
    finally:
        if not installed:
            shutil.rmtree(tmp, ignore_errors=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch pinned third-party source into third_party/."
    )
    parser.add_argument(
        "--force", action="store_true", help="re-fetch every dependency from scratch"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="print each dependency and its status, fetch nothing",
    )
    args = parser.parse_args(argv)

    try:
        validate_manifest(REPOS)
    except ValueError as exc:
        print(f"get_deps.py: {exc}", file=sys.stderr)
        return 1

    if shutil.which("git") is None:
        print("get_deps.py: `git` was not found on PATH.", file=sys.stderr)
        return 1

    try:
        if args.list_only:
            for repo in REPOS:
                print(f"{repo.name}  [{status(repo)}]")
                print(f"  {repo.url}")
                print(f"  {repo.commit}")
                print(f"  {repo.comment}")
            return 0

        for repo in REPOS:
            state = status(repo)
            if not args.force:
                if state == "up to date":
                    print(f"{repo.name}: up to date at {repo.commit[:12]}")
                    continue
                if state.startswith("locally modified"):
                    print(
                        f"{repo.name}: {state}; leaving it alone — run --force to "
                        f"discard those changes and restore the pinned tree"
                    )
                    continue
            reason = "forced re-fetch" if args.force else state
            print(f"{repo.name}: {reason}; fetching {repo.commit[:12]} from {repo.url}")
            dest = vendor(repo)
            print(f"{repo.name}: vendored into {dest.relative_to(REPO_ROOT)}")
    except VendorError as exc:
        print(f"get_deps.py: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
