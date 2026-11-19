"""Conservative alias uncertainty overlay; never narrows an existing mask."""
from collections import defaultdict
import numpy as np

from geolifeclef.neon_record_contract import AcceptedConceptAssessment,NAMED_RANKS
from geolifeclef.taxonomic_equivalence import accepted_alias_classes
from geolifeclef.taxonomy_bridge import normalized_name
from geolifeclef.taxonomy_contract import accepted_concept


class AliasSafeAssessment(AcceptedConceptAssessment):
    def __init__(self,eligible,taxa,mapping,class_names):
        super().__init__(eligible,taxa,mapping,class_names)
        names={}
        for i,name in enumerate(class_names):names.setdefault(normalized_name(name),[]).append(i)
        groups=accepted_alias_classes(taxa,names)
        self.alias_genera=defaultdict(set);self.alias_families=defaultdict(set)
        for code,row in taxa.items():
            if row.get('dwc:taxonRank') not in NAMED_RANKS:continue
            terminal,_=accepted_concept(code,taxa)
            if terminal is None or terminal.get('dwc:taxonRank')!='species':continue
            candidates=groups.get(terminal['taxonID'],set())
            for record in self.chain(row):
                genus=(record.get('dwc:genus') or '').casefold();family=(record.get('dwc:family') or '').casefold()
                if genus:self.alias_genera[genus].update(candidates)
                if family:self.alias_families[family].update(candidates)

    def chain(self,taxon):
        seen=set();row=taxon
        while row is not None and row.get('taxonID') not in seen:
            yield row;seen.add(row.get('taxonID'))
            target=row.get('acceptedTaxonID')
            if target==row.get('taxonID'):break
            row=self.taxa.get(target)

    def compatible(self,taxon):
        ids=set(super().compatible(taxon).tolist())
        if taxon is None:return np.array(sorted(ids),int)
        records=list(self.chain(taxon))
        genera={(r.get('dwc:genus') or '').casefold() for r in records}-{''}
        if genera:
            for genus in genera:ids.update(self.alias_genera.get(genus,set()))
        else:
            for row in records:ids.update(self.alias_families.get((row.get('dwc:family') or '').casefold(),set()))
        for row in records:
            if row.get('taxonID') in self.mapping:ids.add(self.mapping[row['taxonID']])
        return np.array(sorted(ids),int)
