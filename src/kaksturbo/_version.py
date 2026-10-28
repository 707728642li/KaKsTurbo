"""Public packaging version; frozen engine identities are separate."""
import sys
VERSION = "1.1.0"
ORIGINAL_INTERFACE_VERSION = "1.1.0rc9+schur.1"
ENGINE_MANIFESTS = {'win_amd64': 'e055ef1bab36e52bbcc9b14e4fffd5b1c40609e5f7e05910f29d9a343e03fe71', 'linux_x86_64': '42f4b30343ec6da10f957816099881728be12aa2be1e361e41521aa7d9c897ea'}
ENGINE_MANIFEST_SHA256 = ENGINE_MANIFESTS["win_amd64" if sys.platform == "win32" else "linux_x86_64"]
DEVELOPMENT_ENGINE = "engine"
