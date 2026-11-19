"""Run unchanged T2 independent validators with NPZ arrays loaded once.

The frozen checkpoint validator indexes NPZ entries inside record loops. NumPy
otherwise decompresses an entry on every access. Materialization changes I/O
only; preserve the original algorithm and frozen file instead of editing it.
"""
import argparse
import importlib
from pathlib import Path
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checks', choices=['checkpoints', 'events'])
    args = parser.parse_args()
    original = np.load
    stats = {'npz_files_materialized': 0, 'decoded_array_bytes': 0}

    def resident(*arguments, **kwargs):
        result = original(*arguments, **kwargs)
        if not isinstance(result, np.lib.npyio.NpzFile): return result
        with result:
            arrays = {name: result[name] for name in result.files}
        stats['npz_files_materialized'] += 1
        stats['decoded_array_bytes'] += sum(a.nbytes for a in arrays.values())
        return arrays

    module_name = 'validate_taxonomic_verification_t2' + ('_events' if args.checks == 'events' else '')
    module = importlib.import_module(module_name)
    started = time.monotonic()
    try:
        np.load = resident
        module.main()
    finally:
        np.load = original
    from run_taxonomic_resolution_t1 import sha, save
    source = Path(module.__file__)
    save(module.OUT/('resident_'+args.checks+'.json'), {'passed': True,
        'original_validator_sha256': sha(source), 'validator_file': str(source.relative_to(module.ROOT)),
        'I_O_only_adapter': True, 'elapsed_seconds': time.monotonic()-started, **stats})


if __name__ == '__main__': main()
