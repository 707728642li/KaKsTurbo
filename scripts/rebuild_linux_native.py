"""Build the pinned estimator against Conda's ILP64 OpenBLAS.

This experimental packaging script does not change the estimator source or
its floating-point flags. The engine manifest is produced at build time and
still verified at runtime. Conda's final post-processing is separately tested.
"""
from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


manifest_path = ENGINE / "engine-assets.json"
original = json.loads(manifest_path.read_text(encoding="utf-8"))
original_pin = sha(manifest_path)
pins_path = ROOT / "engine_pins.json"
pins = json.loads(pins_path.read_text(encoding="utf-8"))
if original_pin != pins["linux_x86_64"]:
    raise RuntimeError("Source asset manifest does not match the pinned input")
for name, record in original["files"].items():
    path = (ENGINE / name).resolve()
    path.relative_to(ENGINE.resolve())
    if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
        raise RuntimeError("Changed source input: " + name)

cxx = shlex.split(os.environ.get("CXX", "g++"))
prefix = Path(os.environ["PREFIX"]).resolve()
purelib = Path(sysconfig.get_path("purelib"))
build_prefix = os.environ.get("BUILD_PREFIX", "")


def redact_path(value):
    value = value.replace(str(ENGINE.resolve()), "/engine")
    value = value.replace(str(prefix), "<CONDA_PREFIX>")
    if build_prefix:
        value = value.replace(build_prefix, "<BUILD_PREFIX>")
    return value


def installed_rpath(target):
    installed_parent = purelib / "kaksturbo/_engine_root" / target.relative_to(ENGINE).parent
    relative_lib = os.path.relpath(prefix / "lib", installed_parent)
    if os.path.isabs(relative_lib):
        raise RuntimeError("Could not construct a relocatable Conda library path")
    return "$ORIGIN/" + relative_lib


records = []
for recordfile in sorted(ENGINE.glob("build/*/linux-build.json")):
    record = json.loads(recordfile.read_text(encoding="utf-8"))
    oldcmd = record["command"]
    cmd = list(cxx)
    for arg in oldcmd[1:]:
        marker = next((s for s in ("/engine/", "/_engine_root/") if s in arg), None)
        if marker:
            mapped = (ENGINE / arg.split(marker, 1)[1]).resolve()
            mapped.relative_to(ENGINE.resolve())
            arg = ("-I" if arg.startswith("-I") else "") + str(mapped)
        elif arg.startswith(("/data/", "/home/", "/usr/bin/")):
            raise ValueError("Unmapped historical build path: " + arg)
        cmd.append(arg)
    target = Path(cmd[cmd.index("-o") + 1])
    log = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    # Build paths are provenance, not installation-prefix substitutions.
    log = redact_path(log.decode("utf-8", errors="replace")).encode("utf-8")
    (recordfile.parent / "release-build.log").write_bytes(log)
    # Pre-apply the exact Conda post-processing path. A repeat of the same
    # patchelf operation is byte-idempotent; the full-file manifest therefore
    # survives standard relocation. Verify this again on the final package.
    subprocess.run(["patchelf", "--force-rpath", "--set-rpath", installed_rpath(target), str(target)], check=True)
    previous = sha(recordfile)
    record.update(command=[redact_path(x) for x in cmd],
                  compiler=[redact_path(x) for x in cxx], compiler_version=subprocess.check_output(cxx + ["--version"], text=True).splitlines()[0],
                  dll=redact_path(str(target)), dll_sha256=sha(target),
                  packaging_source_record_sha256=previous,
                  scientific_source_unchanged=True, release_rebuild=True)
    recordfile.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    records.append(str(target.relative_to(ENGINE)))

blas = ENGINE / "envs-runtime/linux/blas"
blas.mkdir(parents=True, exist_ok=True)
bridge = ROOT / "scripts/conda_openblas64_bridge.cpp"
provider = prefix / "lib/libopenblas64_.so.0"
if not bridge.is_file() or not provider.is_file():
    raise RuntimeError("Conda ILP64 OpenBLAS bridge or shared library missing")
target = blas / "libkaks_blas64.so"
cmd = cxx + ["-O3", "-std=c++17", "-march=x86-64", "-mtune=generic",
             "-fno-fast-math", "-ffp-contract=off", "-shared", "-fPIC",
             "-static-libgcc", "-static-libstdc++", str(bridge), "-L" + str(prefix / "lib"),
             "-l:libopenblas64_.so.0", "-o", str(target)]
subprocess.run(cmd, check=True)
rpath = installed_rpath(target)
subprocess.run(["patchelf", "--force-rpath", "--set-rpath", rpath, str(target)], check=True)
records.append(str(target.relative_to(ENGINE)))
build = {"source_manifest_sha256": original_pin, "rebuilt_components": records,
         "algorithm_sources_changed": False,
         "third_party_BLAS": "Conda-managed libopenblas-ilp64, not bundled",
         "blas_rpath": rpath, "precision": "FP64", "fast_math": False,
         "max_iterations": 500, "model_count": 14, "gradient_threshold": 1e-8}
(ENGINE / "release-native-build.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
original["files"] = {p.relative_to(ENGINE).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
                     for p in sorted(ENGINE.rglob("*")) if p.is_file() and p != manifest_path}
external = {}
for soname in ("libopenblas64_.so.0", "libgfortran.so.5", "libquadmath.so.0"):
    actual = (prefix / "lib" / soname).resolve(strict=True)
    relative = actual.relative_to(prefix)
    if relative.parts[0] != "lib" or not actual.is_file():
        raise RuntimeError("Conda numerical dependency escaped the build prefix: " + soname)
    external[relative.as_posix()] = {"bytes": actual.stat().st_size, "sha256": sha(actual)}
original["conda_external_assets"] = external
original["packaging_source_manifest_sha256"] = original_pin
manifest_path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")
pin = sha(manifest_path)
old = pins["linux_x86_64"]
pins["linux_x86_64"] = pin
pins_path.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
version = ROOT / "src/kaksturbo/_version.py"
text = version.read_text(encoding="utf-8")
if old not in text:
    raise RuntimeError("Version module does not contain the source engine pin")
version.write_text(text.replace(old, pin), encoding="utf-8")
identity = ROOT / "release_identity.json"
if identity.is_file():
    data = json.loads(identity.read_text(encoding="utf-8"))
    data["pins"] = pins
    data["native_source_rebuilt"] = True
    identity.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(json.dumps(build))
