"""Build the optional NumPy/f2py extension on Linux with GNU Fortran.

Requires Python development headers, NumPy, gcc and gfortran. Uses f2py's
wrapper generator directly, so Meson/Ninja are not required. FC, FCFLAGS,
CC, CFLAGS, LDFLAGS and FCLIBS may be overridden in the environment.
This script compiles local sources only; it installs no packages.
"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

import numpy as np
import numpy.f2py


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "_build_python"


def setting(name, default):
    return shlex.split(os.environ.get(name, default))


def run(argv):
    print(shlex.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=BUILD, check=True)


def main():
    if not sys.platform.startswith("linux"):
        raise SystemExit("This build helper targets Linux; use hhoo_pes.pyf with your platform's f2py build.")
    BUILD.mkdir(exist_ok=True)
    fc, cc = setting("FC", "gfortran"), setting("CC", "gcc")
    fcflags = setting("FCFLAGS", "-O3 -frecursive")
    module_flag = setting("FC_MODULE_FLAG", "-J")
    cflags = setting("CFLAGS", "-O3")
    ldflags = setting("LDFLAGS", "")
    fclibs = setting("FCLIBS", "-lgfortran -lm")
    run([sys.executable, "-m", "numpy.f2py", ROOT / "hhoo_pes.pyf"])
    objects = []
    for source in [ROOT / "hhoo_1014_analytic.f90", *sorted(BUILD.glob("hhoo_pes-f2pywrappers*.f*"))]:
        if source.stat().st_size == 0:
            continue
        obj = BUILD / (source.stem + ".o")
        run([*fc, *fcflags, "-fPIC", *module_flag, BUILD, "-I", BUILD, "-c", source, "-o", obj])
        objects.append(obj)
    f2py_include = Path(numpy.f2py.get_include())
    includes = ["-I" + sysconfig.get_path("include"), "-I" + np.get_include(), "-I" + str(f2py_include)]
    for source in (BUILD / "hhoo_pesmodule.c", f2py_include / "fortranobject.c"):
        obj = BUILD / (source.stem + ".o")
        run([*cc, *cflags, "-fPIC", *includes, "-c", source, "-o", obj])
        objects.append(obj)
    output = ROOT / ("hhoo_pes" + sysconfig.get_config_var("EXT_SUFFIX"))
    run([*cc, "-shared", *ldflags, *objects, *fclibs, "-o", output])
    print("Built:", output)


if __name__ == "__main__":
    main()
