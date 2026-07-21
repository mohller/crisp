"""Download external data dependencies from GitHub.

The CRPropa3-data repository (https://github.com/CRPropa/CRPropa3-data)
contains photodisintegration and photopion cross-section tables, and the
AstroPhoMes repository (https://github.com/mohller/AstroPhoMes) provides the
photomeson models usable through crisp.photonuclear_cross_sections.Photomeson.

Public API
----------
fetch_crpropa_tables(destination, tables, verbose)
    Download table sub-directories from GitHub and cache them locally.

get_tables_path(destination, auto_download, verbose)
    Resolve the local tables path, downloading if necessary.  Used
    internally by InteractionCore classes.

fetch_astrophomes(destination, verbose)
    Download the AstroPhoMes repository and cache it locally.

get_astrophomes_path(destination, auto_download, verbose)
    Resolve the local AstroPhoMes path, downloading if necessary.  Used by
    crisp.photonuclear_cross_sections.load_astrophomes.
"""

import io
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_REPO_ZIP = "https://github.com/CRPropa/CRPropa3-data/archive/refs/heads/master.zip"
_DEFAULT_TABLES = ["PD_Talys1.8_Khan", "PD_Talys1.9", "PD_external", "PPP"]
_CACHE_DIR = Path.home() / ".cache" / "crisp" / "CRPropa3-data" / "tables"

_ASTROPHOMES_ZIPS = [
    "https://github.com/mohller/AstroPhoMes/archive/refs/heads/master.zip",
    "https://github.com/mohller/AstroPhoMes/archive/refs/heads/main.zip",
]
_ASTROPHOMES_CACHE = Path.home() / ".cache" / "crisp" / "AstroPhoMes"

# Everything load_astrophomes actually needs: it execs config.py and then
# imports photomeson_lib.photomeson_models. Testing only config.py would call
# an interrupted download "complete" -- config.py sorts before photomeson_lib/
# in the archive, so a download cut off in between leaves a cache that looks
# valid forever and fails with ModuleNotFoundError: No module named
# 'photomeson_lib'.
_ASTROPHOMES_REQUIRED = ("config.py", "photomeson_lib/photomeson_models.py")


def _astrophomes_complete(path):
    """True when *path* holds a usable AstroPhoMes checkout, not a partial one."""
    return all(Path(path, rel).exists() for rel in _ASTROPHOMES_REQUIRED)


def fetch_crpropa_tables(destination=None, tables=None, verbose=True):
    """Download CRPropa3 cross-section tables from GitHub into a local cache.

    Downloads the CRPropa3-data repository as a zip archive and extracts
    only the requested table sub-directories.  Files that are already
    present locally are skipped.

    Parameters
    ----------
    destination : str | Path | None
        Local directory for the tables.
        Defaults to ``~/.cache/crisp/CRPropa3-data/tables/``.
    tables : list[str] | None
        Sub-directory names (under ``tables/``) to extract.
        Defaults to ``["PD_Talys1.8_Khan", "PD_Talys1.9", "PD_external", "PPP"]``.
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        Path to the local tables directory with a trailing ``/``.

    Examples
    --------
    >>> import crisp
    >>> path = crisp.fetch_crpropa_tables()
    >>> ic = InteractionCore_UHECR_Source(data_directory=path, ...)
    """
    dest = Path(destination) if destination else _CACHE_DIR
    if tables is None:
        tables = _DEFAULT_TABLES

    missing = [t for t in tables if not (dest / t).exists()]
    if not missing:
        return str(dest) + "/"

    if verbose:
        print(f"Downloading CRPropa3 tables ({', '.join(missing)}) from GitHub …")

    resp = urllib.request.urlopen(_REPO_ZIP)
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        prefix = "CRPropa3-data-master/tables/"
        for name in zf.namelist():
            for table in missing:
                if name.startswith(f"{prefix}{table}/") and not name.endswith("/"):
                    rel = name[len(prefix):]
                    out = dest / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(name))

    if verbose:
        print(f"Tables saved to: {dest}")
    return str(dest) + "/"


def get_tables_path(destination=None, auto_download=True, verbose=True):
    """Return the local path to CRPropa3 cross-section tables.

    Resolution order:

    1. ``CRPROPA_TABLES_PATH`` environment variable.
    2. *destination* argument, if the directory already exists.
    3. Default cache (``~/.cache/crisp/CRPropa3-data/tables/``), if populated.
    4. Auto-download from GitHub (when ``auto_download=True``).

    Parameters
    ----------
    destination : str | Path | None
        Explicit path override.
    auto_download : bool
        Fetch tables from GitHub if not found locally.
    verbose : bool
        Print progress during download.

    Returns
    -------
    str
        Resolved path with a trailing ``/``.

    Raises
    ------
    FileNotFoundError
        When the tables are not found and ``auto_download`` is *False*.
    """
    env = os.environ.get("CRPROPA_TABLES_PATH", "")
    if env:
        return env if env.endswith("/") else env + "/"

    if destination is not None:
        p = Path(destination)
        if p.exists():
            return str(p) + "/"

    if _CACHE_DIR.exists() and any(_CACHE_DIR.iterdir()):
        return str(_CACHE_DIR) + "/"

    if auto_download:
        return fetch_crpropa_tables(destination=destination, verbose=verbose)

    raise FileNotFoundError(
        "CRPropa3 tables not found.\n"
        "Run crisp.fetch_crpropa_tables() to download them, or set the\n"
        "CRPROPA_TABLES_PATH environment variable to an existing directory."
    )


def fetch_astrophomes(destination=None, verbose=True):
    """Download the AstroPhoMes repository from GitHub into a local cache.

    The archive top-level directory is stripped, so the cache directory is
    the repository root (containing ``config.py`` and ``photomeson_lib/``).
    Skipped when the destination already holds a copy.

    Parameters
    ----------
    destination : str | Path | None
        Local directory for the repository.
        Defaults to ``~/.cache/crisp/AstroPhoMes/``.
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        Path to the local repository root.
    """
    dest = Path(destination) if destination else _ASTROPHOMES_CACHE

    if _astrophomes_complete(dest):
        return str(dest)

    # A leftover partial copy would otherwise be indistinguishable from a good
    # one for whichever files it does contain; start from a clean slate.
    if dest.exists():
        if verbose:
            print(f"Incomplete AstroPhoMes copy at {dest} — re-downloading.")
        shutil.rmtree(dest, ignore_errors=True)

    last_error = None
    for url in _ASTROPHOMES_ZIPS:
        staging = None
        try:
            if verbose:
                print(f"Downloading AstroPhoMes from {url} …")
            resp = urllib.request.urlopen(url)
            # Unpack into a sibling staging directory and only move it into
            # place once it is complete, so an interruption (dropped network,
            # full disk, Ctrl-C) can never leave a half-populated cache that
            # later runs mistake for a finished download.
            dest.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=dest.name + ".part-",
                                            dir=dest.parent))
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                names = zf.namelist()
                prefix = names[0].split("/")[0] + "/"
                for name in names:
                    if name.endswith("/") or not name.startswith(prefix):
                        continue
                    out = staging / name[len(prefix):]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(name))

            if not _astrophomes_complete(staging):
                missing = [r for r in _ASTROPHOMES_REQUIRED
                           if not (staging / r).exists()]
                raise RuntimeError(
                    f"archive from {url} is missing {', '.join(missing)}")

            os.replace(staging, dest)
            staging = None
            if verbose:
                print(f"AstroPhoMes saved to: {dest}")
            return str(dest)
        except Exception as exc:            # try the next branch name
            last_error = exc
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    raise RuntimeError(f"Could not download AstroPhoMes: {last_error}")


def get_astrophomes_path(destination=None, auto_download=True, verbose=True):
    """Return the local path to the AstroPhoMes repository.

    Resolution order:

    1. ``ASTROPHOMES_PATH`` environment variable.
    2. *destination* argument, if the directory already exists.
    3. Default cache (``~/.cache/crisp/AstroPhoMes/``), if populated.
    4. Auto-download from GitHub (when ``auto_download=True``).

    Returns
    -------
    str
        Resolved repository root.

    Raises
    ------
    FileNotFoundError
        When the repository is not found and ``auto_download`` is *False*.
    """
    env = os.environ.get("ASTROPHOMES_PATH", "")
    if env and _astrophomes_complete(env):
        return env

    if destination is not None and _astrophomes_complete(destination):
        return str(destination)

    if _astrophomes_complete(_ASTROPHOMES_CACHE):
        return str(_ASTROPHOMES_CACHE)

    if auto_download:
        return fetch_astrophomes(destination=destination, verbose=verbose)

    raise FileNotFoundError(
        "No complete AstroPhoMes repository found.\n"
        f"A copy counts as complete only when it contains "
        f"{' and '.join(_ASTROPHOMES_REQUIRED)}, so a partially downloaded\n"
        "cache is reported as missing rather than used.\n"
        "Run crisp.fetch_astrophomes() to download it, or set the\n"
        "ASTROPHOMES_PATH environment variable to an existing working copy."
    )
