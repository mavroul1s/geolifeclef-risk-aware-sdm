"""A1 metadata audit; no plot composition, F1 outcomes, refits or label edits."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from geolifeclef.taxonomy_contract import (INFRASPECIFIC, accepted_concept, parent_candidates,
                                          species_catalog_condition, species_name_index)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    out = ROOT/'results/neon_taxonomy_a1'; out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():
        raise ValueError('A1 already complete; preserve this audit')
    receipt = json.loads((ROOT/'results/neon_bridge/taxonomy_retrieval.json').read_text())
    taxa = {}
    for record in receipt['files']:
        path = ROOT/record['file']; assert sha(path) == record['sha256']
        for row in json.loads(path.read_text())['data']:
            assert row['taxonID'] not in taxa
            taxa[row['taxonID']] = row
    assert len(taxa) == receipt['expected_dictionary_entries']
    mapping_table = pd.read_csv(ROOT/'data/neon_bridge/neon_to_sage_taxonomy.csv')
    mapping = {r.neon_taxon_id: int(r.sage_index) for r in mapping_table[mapping_table.sage_index.notna()].itertuples()}
    contract = np.load(ROOT/'data/confirmation_e4/label_contract.npz')
    cells = pd.read_csv(ROOT/'data/confirmation_e4/neon_assessment_roles.csv')
    names = pd.read_csv(ROOT/'data/neon_bridge/species_names.csv')['Species Name']
    genera = names.str.split().str[0].str.casefold().to_numpy()
    index = species_name_index(taxa)
    chains, parents = Counter(), Counter(); candidates = []
    accepted_species, accepted_infra = defaultdict(list), Counter()
    for code, row in taxa.items():
        terminal, status = accepted_concept(code, taxa); chains[status] += 1
        if row.get('dwc:taxonRank') in INFRASPECIFIC:
            result = parent_candidates(row, taxa, index); parents[result['status']] += 1
            candidates.append({'taxon_id': code, 'status': result['status'], 'candidate_count': len(result['parent_ids']),
                'candidate_parent_ids': '|'.join(result['parent_ids']),
                'candidate_has_existing_model_mapping': any(i in mapping for i in result['parent_ids'])})
        if terminal is not None and terminal['taxonID'] == code:
            genus = (row.get('dwc:genus') or '').casefold()
            if row.get('dwc:taxonRank') == 'species':
                accepted_species[genus].append(code)
            elif row.get('dwc:taxonRank') in INFRASPECIFIC:
                accepted_infra[genus] += 1
    # Dictionary-only candidates stay local; summaries are public research output.
    data = ROOT/'data/neon_taxonomy_a1'; data.mkdir(exist_ok=True)
    pd.DataFrame(candidates).to_csv(data/'parent_candidates.csv', index=False)
    rows = []
    for i, cell in enumerate(cells.itertuples()):
        eligible = contract['neon_eligible'][i]
        for genus in np.unique(genera[eligible]):
            result = species_catalog_condition(accepted_species[genus], mapping, eligible)
            rows.append({'cell_id': cell.cell_id, 'genus': genus, **result,
                'accepted_infraspecific_concepts_requiring_separate_contract': accepted_infra[genus],
                'field_positive_genus_guarantee_established': False})
    screen = pd.DataFrame(rows); screen.to_csv(out/'species_catalog_screen.csv', index=False)
    result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'metadata_audit_complete_field_contract_partial',
        'dictionary_entries': len(taxa), 'dictionary_pages_verified': len(receipt['files']),
        'accepted_chain_status': dict(chains), 'infraspecific_dictionary_status': dict(parents),
        'cell_genus_pairs_screened': len(screen), 'candidate_cells': len(cells),
        'pairs_passing_necessary_species_catalog_condition': int(screen.necessary_species_catalog_condition.sum()),
        'pairs_with_unmapped_catalog_species': int(screen.unmapped_concepts.gt(0).sum()),
        'pairs_with_mapped_outside_eligible_species': int(screen.mapped_outside_eligible.gt(0).sum()),
        'pairs_with_unresolved_infraspecific_contract': int(screen.accepted_infraspecific_concepts_requiring_separate_contract.gt(0).sum()),
        'field_positive_genus_guarantees_established': 0,
        'new_plot_composition_reads': 0, 'new_F1_evaluations': 0, 'model_or_label_changes': False,
        'history_semantics': 'current published records retain current determinations; identificationHistory archives prior determinations, not a replacement-label source',
        'limits': ['acceptedTaxonID is accepted-name linkage, not an infraspecific parent key',
            'lexical species-parent candidates require validation; no automatic rank collapse',
            'unmapped named concepts remain unresolved, not certified outside-universe absences',
            'species-only catalog closure is a necessary screen, not complete botanical or field observation coverage',
            'protected/higher-rank identifications are not reverse-resolved',
            'selected-key history and species-parent harmonization remain pending; E4 outcomes remain consumed'],
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT/'results/neon_bridge/taxonomy_retrieval.json', ROOT/'data/neon_bridge/neon_to_sage_taxonomy.csv',
            ROOT/'data/confirmation_e4/label_contract.npz', ROOT/'data/neon_bridge/BARRreadme.txt', ROOT/'data/neon_bridge/BARRvariables.csv']}}
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
