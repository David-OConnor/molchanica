from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
web = root.parent / 'bio_web'
lib = root.parent / 'bio_tools'
stage = root / '.tool-parity-stage'
package = stage / 'bio_tools/src/adapters/python/bio_tool_adapters'
package.mkdir(parents=True, exist_ok=True)
names = ['rfdiffusion3', 'proteinmpnn', 'ligandmpnn', 'opendde', 'boltz2', 'chai1', 'esmfold2']
common = (web / 'main/tools/__init__.py').read_text(encoding='utf-8')
registry = common[common.index('def all_tools()'):common.index('def _bundled_candidates')]
shared = common[:common.index('# todo: Can this be rolled')] + common[common.index('def _bundled_candidates'):common.index('from . import (  # noqa: E402')]
(package / '__init__.py').write_text(shared, encoding='utf-8')
for name in names + ['field_processing', 'environments', 'status_check']:
    source = (web / f'main/tools/{name}.py').read_text(encoding='utf-8')
    if name == 'field_processing':
        source = source.replace('from main.tools import', 'from . import')
    if name == 'environments':
        source = source.replace('os.getenv("BIO_WEB_EXECUTABLE_ROOT")', 'os.getenv("BIO_TOOLS_EXECUTABLE_ROOT") or os.getenv("BIO_WEB_EXECUTABLE_ROOT")')
        source = source.replace('Path(__file__).resolve().parents[2] / "process_executables"', 'Path.cwd() / "process_executables"')
    source = source.replace('Path(__file__).resolve().parents[2] / "tool_scripts"', 'Path(__file__).resolve().parent / "tool_scripts"')
    source = source.replace('Path(__file__).resolve().parents[2] / "tool_data"', 'Path(__file__).resolve().parent / "tool_data"')
    (package / f'{name}.py').write_text(source, encoding='utf-8')
(package / 'tool_scripts').mkdir(exist_ok=True)
shutil.copy2(web / 'tool_scripts/esmfold2_inference.py', package / 'tool_scripts/esmfold2_inference.py')
shutil.copytree(web / 'tool_data/chai1', package / 'tool_data/chai1', dirs_exist_ok=True)

# Preserve web-only registry and late imports; all runner helpers have one owner.
web_init = '''"""Web registry backed by the shared bio_tools adapters."""
import os
import sys
from pathlib import Path
import bio_tools

os.environ.setdefault("BIO_WEB_EXECUTABLE_ROOT", str(Path(__file__).resolve().parents[2] / "process_executables"))
_adapter_path = bio_tools.adapter_package_path()
if _adapter_path not in sys.path:
    sys.path.insert(0, _adapter_path)
import bio_tool_adapters as _shared
globals().update({name: value for name, value in vars(_shared).items() if not name.startswith("__")})

'''
dest = stage / 'bio_web/main/tools'
dest.mkdir(parents=True, exist_ok=True)
(dest / '__init__.py').write_text(web_init + registry + common[common.index('from . import (  # noqa: E402'):], encoding='utf-8')
for name in names + ['field_processing', 'environments', 'status_check']:
    (dest / f'{name}.py').write_text(f'"""Compatibility import for the shared bio_tools implementation."""\nimport sys\nfrom bio_tool_adapters import {name} as _implementation\nsys.modules[__name__] = _implementation\n', encoding='utf-8')

rust = '''//! Shared scientific adapters used by desktop and web clients.
//!
//! Python owns input validation and native command construction; execution and
//! durable artifacts use the bio_tools Python binding's CommandSpec.
use std::{fs, io, path::{Path, PathBuf}, hash::{Hash, Hasher}};

const FILES: &[(&str, &[u8])] = &[
'''
for file in sorted(package.rglob('*')):
    if file.is_file():
        rel = file.relative_to(package.parent).as_posix()
        rust += f'    ("{rel}", include_bytes!("python/{rel}")),\n'
rust += '''];

/// Materialize the versioned adapter package, including its licensed example assets.
pub fn package_path() -> io::Result<PathBuf> {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    for (name, data) in FILES { name.hash(&mut hash); data.hash(&mut hash); }
    let root = std::env::temp_dir().join(format!("bio-tools-adapters-{:x}", hash.finish()));
    write_package(&root)?;
    Ok(root)
}

pub fn write_package(root: &Path) -> io::Result<()> {
    for (name, data) in FILES {
        let path = root.join(name);
        if fs::read(&path).ok().as_deref() == Some(*data) { continue; }
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, data)?;
    }
    Ok(())
}
'''
(stage / 'bio_tools/src/adapters/mod.rs').write_text(rust, encoding='utf-8')
(stage / 'bio_tools/src/lib.rs').write_text((lib / 'src/lib.rs').read_text(encoding='utf-8').replace('mod input;', 'pub mod adapters;\nmod input;'), encoding='utf-8')
native = (lib / 'python/src/lib.rs').read_text(encoding='utf-8')
native = native.replace('#[pymodule]\nfn bio_tools', '''/// Location of the embedded, versioned scientific adapter package.
#[pyfunction]
fn adapter_package_path() -> PyResult<String> {
    bio_tools_rs::adapters::package_path()
        .map(|path| path.to_string_lossy().into_owned())
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

#[pymodule]
fn bio_tools''')
native = native.replace('metadata::register(m)?;', 'metadata::register(m)?;\n    m.add_function(wrap_pyfunction!(adapter_package_path, m)?)?;')
(stage / 'bio_tools/python/src').mkdir(parents=True, exist_ok=True)
(stage / 'bio_tools/python/src/lib.rs').write_text(native, encoding='utf-8')
print('Staged shared adapters and web compatibility imports:', stage)
