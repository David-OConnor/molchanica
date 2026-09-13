"""File-based desktop entry point for the same payloads accepted by the web API."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import traceback
import zipfile

from . import preset_payload

SUPPORTED = {
    'rfd3': 'rfdiffusion3', 'proteinmpnn': 'proteinmpnn',
    'ligandmpnn': 'ligandmpnn', 'opendde': 'opendde',
    'boltz2': 'boltz2', 'chai1': 'chai1', 'esmfold2': 'esmfold2',
}


def main() -> None:
    slug, request_file, response_file = sys.argv[1:]
    response = Path(response_file)
    try:
        if slug not in SUPPORTED:
            raise ValueError(f'Unsupported desktop adapter: {slug}')
        adapter = importlib.import_module(f'.{SUPPORTED[slug]}', __package__)
        payload = json.loads(Path(request_file).read_text(encoding='utf-8'))
        result = adapter.run(preset_payload(slug, payload))
        log = Path(result['run_log_dir'])
        archive = response.parent / 'raw-results.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
            for file in sorted(log.rglob('*')):
                if file.is_file():
                    output.write(file, file.relative_to(log))
            output.write(request_file, 'submitted-form.json')
        result['archive'] = str(archive)
        response.write_text(json.dumps({'result': result}, indent=2), encoding='utf-8')
    except Exception as error:
        response.write_text(json.dumps({'error': str(error)}, indent=2), encoding='utf-8')
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
