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

    if (dest / "config.py").exists():
        return str(dest)

    last_error = None
    for url in _ASTROPHOMES_ZIPS:
        try:
            if verbose:
                print(f"Downloading AstroPhoMes from {url} …")
            resp = urllib.request.urlopen(url)
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                names = zf.namelist()
                prefix = names[0].split("/")[0] + "/"
                for name in names:
                    if name.endswith("/") or not name.startswith(prefix):
                        continue
                    out = dest / name[len(prefix):]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(name))
            if verbose:
                print(f"AstroPhoMes saved to: {dest}")
            return str(dest)
        except Exception as exc:            # try the next branch name
            last_error = exc

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
    if env and Path(env, "config.py").exists():
        return env

    if destination is not None and Path(destination, "config.py").exists():
        return str(destination)

    if (_ASTROPHOMES_CACHE / "config.py").exists():
        return str(_ASTROPHOMES_CACHE)

    if auto_download:
        return fetch_astrophomes(destination=destination, verbose=verbose)

    raise FileNotFoundError(
        "AstroPhoMes repository not found.\n"
        "Run crisp.fetch_astrophomes() to download it, or set the\n"
        "ASTROPHOMES_PATH environment variable to an existing working copy."
    )
