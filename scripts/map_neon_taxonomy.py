"""Map public dictionary identifiers without inspecting NEON plot composition."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geolifeclef.taxonomy_bridge import map_taxon, normalized_name


def main():
    data, out = ROOT / 'data/neon_bridge', ROOT / 'results/neon_bridge'
    receipt = json.loads((out / 'taxonomy_retrieval.json').read_text())
    taxa = []
    for file in receipt['files']:
        body = (ROOT / file['file']).read_bytes()
        if hashlib.sha256(body).hexdigest() != file['sha256']:
            raise ValueError('Taxonomy dictionary digest changed')
        taxa.extend(json.loads(body)['data'])
    by_id = {r['taxonID']: r for r in taxa}
    if len(by_id) != len(taxa) or len(taxa) != receipt['expected_dictionary_entries']:
        raise ValueError('Duplicate or missing taxon IDs')
    species_file = data / 'species_names.csv'
    species = pd.read_csv(species_file)
    if list(species.Index) != list(range(5771)):
        raise ValueError('SAGE class-order mismatch')
    names = {}
    for _, row in species.iterrows():
        names.setdefault(normalized_name(row['Species Name']), []).append(int(row.Index))
    rows = []
    for taxon in taxa:
        rows.append({'neon_taxon_id': taxon['taxonID'], 'neon_accepted_taxon_id': taxon.get('acceptedTaxonID'),
                     'neon_rank': taxon.get('dwc:taxonRank'), **map_taxon(taxon, by_id, names)})
    frame = pd.DataFrame(rows)
    frame['sage_index'] = frame.sage_index.astype('Int64')
    path = data / 'neon_to_sage_taxonomy.csv'
    frame.to_csv(path, index=False)
    result = {'status': 'dictionary_mapping_complete', 'dictionary_entries': len(frame),
              'status_counts': {k: int(v) for k, v in frame.status.value_counts().items()},
              'distinct_SAGE_classes_matched': int(frame.sage_index.nunique()), 'SAGE_classes_total': 5771,
              'observation_weighted_coverage_computed': False, 'NEON_species_occurrences_read': False,
              'no_automatic_infraspecific_or_higher_rank_collapse': True,
              'mapping_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'sage_class_file_sha256': hashlib.sha256(species_file.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256((ROOT / 'src/geolifeclef/taxonomy_bridge.py').read_bytes()).hexdigest(),
              'warning': 'Dictionary-level matches do not measure observed taxonomic coverage, native masks, or complete absence semantics.'}
    (out / 'taxonomy_mapping.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
