"""Selected-key provenance and conservative current-record concept handling."""
from datetime import datetime, timezone
import re
import numpy as np

from geolifeclef.neon_assessment import TaxonomicAssessment
from geolifeclef.neon_morphospecies import csv_byte_records, latest_resolution
from geolifeclef.taxonomy_bridge import canonical_name
from geolifeclef.taxonomy_contract import INFRASPECIFIC, accepted_concept

NAMED_RANKS = {'species'} | INFRASPECIFIC


class AcceptedConceptAssessment(TaxonomicAssessment):
    """Unknown masks follow full accepted chains and retain fallback uncertainty."""
    def compatible(self, taxon):
        if taxon is None:
            return np.arange(self.eligible.shape[1])
        records=[];seen=set();row=taxon
        while row is not None and row.get('taxonID') not in seen:
            records.append(row);seen.add(row.get('taxonID'))
            target=row.get('acceptedTaxonID')
            if target==row.get('taxonID'):
                break
            row=self.taxa.get(target)
        genera={(r.get('dwc:genus') or '').casefold() for r in records}-{''}
        ids=set().union(*(self.genera.get(g,set()) for g in genera)) if genera else set()
        if ids:
            return np.array(sorted(ids),dtype=int)
        families={(r.get('dwc:family') or '').casefold() for r in records}-{''}
        if families:
            ids=set().union(*(self.families.get(f,set()) for f in families))
            ids|=set(range(self.eligible.shape[1]))-self.classes_with_family
            if ids:
                return np.array(sorted(ids),dtype=int)
        return np.arange(self.eligible.shape[1])


def selected_key_rows(payload, keys, key_columns, allowed, identity_keys=None, identity_fields=()):
    """Decode selected keys only; sensitive identity fields need a second guard."""
    records = iter(csv_byte_records(payload))
    header = [v.decode('utf-8-sig') for v in next(records)]
    if len(set(header)) != len(header):
        raise ValueError('Duplicate columns')
    positions = [header.index(k) for k in key_columns]
    for record in records:
        if len(record) != len(header):
            raise ValueError('CSV row width mismatch')
        key = tuple(record[i].decode('utf-8') for i in positions)
        if key not in keys:
            continue
        fields = set(allowed) | set(key_columns)
        if identity_keys is not None and key in identity_keys:
            fields.update(identity_fields)
        yield {name: record[i].decode('utf-8') for i, name in enumerate(header) if name in fields}


def accepted_species_mapping(taxon, taxa, sage_names):
    """Only exact species concepts and explicit accepted-species links qualify."""
    rank = taxon.get('dwc:taxonRank')
    if rank not in NAMED_RANKS:
        return {'sage_index': None, 'status': 'higher_rank_or_unknown'}
    terminal, status = accepted_concept(taxon['taxonID'], taxa)
    if terminal is None:
        return {'sage_index': None, 'status': status}
    candidates = set()
    if rank == 'species':
        candidates.update(sage_names.get(canonical_name(taxon), []))
    if terminal.get('dwc:taxonRank') == 'species':
        candidates.update(sage_names.get(canonical_name(terminal), []))
    if len(candidates) != 1:
        return {'sage_index': None, 'status': 'conflicting_species_candidates' if candidates else 'unmatched_named_concept'}
    return {'sage_index': next(iter(candidates)), 'status': 'exact_species' if rank == 'species' else 'explicit_accepted_species'}


def public_rank(row, taxa):
    return (row.get('taxonRank') or taxa.get(row.get('taxonID'), {}).get('dwc:taxonRank') or '').casefold()


def named_concept(row, taxa):
    taxon = taxa.get(row.get('taxonID'))
    if taxon is None or public_rank(row, taxa) not in NAMED_RANKS:
        return None
    terminal, _ = accepted_concept(taxon['taxonID'], taxa)
    return terminal['taxonID'] if terminal is not None else None


def rank_matches(row, taxa):
    taxon = taxa.get(row.get('taxonID'))
    if taxon is None:
        return True  # Missing dictionary identity remains unknown downstream.
    declared = row.get('taxonRank', '').strip().casefold()
    return not declared or declared == (taxon.get('dwc:taxonRank') or '').casefold()


def current_resolution(rows):
    chosen, status = latest_resolution(rows)
    if status != 'conflicting_latest_resolution':
        return chosen, status, []
    dated = []
    for row in rows:
        try:
            date = datetime.fromisoformat(row.get('morphospeciesResolvedDate', '').replace('Z', '+00:00'))
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            dated.append((date, row))
        except ValueError:
            pass
    newest = max(d for d, _ in dated)
    return None, status, [r for d, r in dated if d == newest]


def combine_current_record(original, resolution, status, conflicts, taxa):
    """Return chosen row, reason, uncertain evidence; history is not an input."""
    if status == 'conflicting_latest_resolution':
        return dict(original), 'ambiguous_current_resolution', [original, *conflicts]
    if resolution is None:
        return dict(original), 'original_current_record', []
    old, new = named_concept(original, taxa), named_concept(resolution, taxa)
    if old is not None and new is not None and old != new:
        return dict(original), 'conflicting_current_named_concepts', [original, resolution]
    chosen = dict(original)
    for field in ['taxonID', 'taxonRank', 'scientificName', 'identificationQualifier', 'dataQF']:
        chosen[field] = resolution.get(field, '')
    if original.get('dataQF', '').strip() not in {'', '0', '0.0'}:
        chosen['dataQF'] = original['dataQF']
    return chosen, 'published_current_resolution', []


def history_table_matches(row, table):
    tokens = re.findall(r'div_[A-Za-z0-9_]+', row.get('identificationTable', ''))
    return table in {t.removesuffix('_pub') for t in tokens}
