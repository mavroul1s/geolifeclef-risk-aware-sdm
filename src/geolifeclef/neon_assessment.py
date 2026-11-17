"""Selected-record access, conservative taxonomic missingness and fixed-set scoring."""
from collections import Counter
import numpy as np


def selected_csv_rows(payload, selected, allowed):
    """Decode composition fields only after plot/bout identity is admitted.

    Unselected outcome bytes can be invalid UTF-8: they never undergo decoding.
    CSV quotes, escaped quotes and embedded newlines are parsed at byte level.
    """
    data=payload if payload.endswith(b'\n') else payload+b'\n'
    header=None;capture_columns=None;fields={};value=bytearray()
    column=position=0;quoted=False;field_start=True
    while position<len(data):
        char=data[position];capture=header is None or column in capture_columns
        if char==34:
            if quoted:
                if position+1<len(data) and data[position+1]==34:
                    if capture:value.append(34)
                    position+=2;continue
                quoted=False
            elif field_start:quoted=True
            else:raise ValueError('Unexpected CSV quote')
            field_start=False
        elif not quoted and char in (44,10,13):
            if capture:fields[column]=bytes(value)
            value.clear();column+=1;field_start=True
            if char in (10,13):
                if char==13 and position+1<len(data) and data[position+1]==10:position+=1
                if header is None:
                    header=[fields[i].decode('utf-8-sig') for i in range(column)]
                    if len(set(header))!=len(header) or not {'plotID','boutNumber','taxonID'}.issubset(header):raise ValueError('CSV identity/schema mismatch')
                    capture_columns=[i for i,name in enumerate(header) if name in set(allowed)|{'plotID','boutNumber'}]
                    plot_column,bout_column=header.index('plotID'),header.index('boutNumber')
                else:
                    if column!=len(header):raise ValueError('CSV row width mismatch')
                    plot=fields[plot_column].decode('utf-8');bout=fields[bout_column].decode('utf-8')
                    if selected.get(plot)==bout:
                        yield {header[i]:fields[i].decode('utf-8') for i in capture_columns}
                fields={};column=0
        else:
            if capture:value.append(char)
            field_start=False
        position+=1
    if quoted:raise ValueError('Unclosed CSV quote')


class TaxonomicAssessment:
    def __init__(self, eligible, taxa, mapping, class_names):
        self.eligible=np.asarray(eligible,bool)
        self.positive=np.zeros_like(self.eligible);self.uncertain=np.zeros_like(self.eligible)
        self.taxa=taxa;self.mapping=mapping;self.counts=Counter()
        self.genera={};self.families={}
        for i,name in enumerate(class_names):self.genera.setdefault(name.split()[0].casefold(),set()).add(i)
        self.classes_with_family=set()
        for code,index in mapping.items():
            family=(taxa.get(code,{}).get('dwc:family') or '').casefold()
            if family:self.families.setdefault(family,set()).add(index);self.classes_with_family.add(index)

    def compatible(self,taxon):
        if taxon is None:return np.arange(self.eligible.shape[1])
        accepted=self.taxa.get(taxon.get('acceptedTaxonID'),taxon)
        genera={str(t.get('dwc:genus') or '').casefold() for t in [taxon,accepted]}-{''}
        if genera:return np.array(sorted(set().union(*(self.genera.get(g,set()) for g in genera))),dtype=int)
        families={str(t.get('dwc:family') or '').casefold() for t in [taxon,accepted]}-{''}
        if families:
            ids=set().union(*(self.families.get(f,set()) for f in families))
            ids|=set(range(self.eligible.shape[1]))-self.classes_with_family
            return np.array(sorted(ids),dtype=int)
        return np.arange(self.eligible.shape[1])

    def add(self,cell,row,table):
        if table=='div_1m2Data' and row.get('divDataType','').casefold()!='plantspecies':return
        self.counts['selected_plant_records']+=1
        code=row.get('taxonID','').strip();present=row.get('targetTaxaPresent','').strip().casefold()
        yes=present in {'yes','y','true','1'}
        if not code:
            if yes:
                self.uncertain[cell]=True;self.counts['positive_without_taxon_id']+=1
            elif present not in {'','no','n','false','0'}:raise ValueError('Unknown targetTaxaPresent code')
            return
        if not yes:raise ValueError('Taxon record without affirmative presence flag')
        qualifier=row.get('identificationQualifier','').strip().casefold()
        quality=row.get('dataQF','').strip()
        resolved=qualifier in {'','na','none','no qualifier'} and quality in {'','0','0.0'}
        index=self.mapping.get(code)
        if index is not None and resolved:
            self.counts['mapped_resolved_records']+=1
            if self.eligible[cell,index]:self.positive[cell,index]=True
            else:self.counts['mapped_outside_native_universe']+=1
        else:
            self.counts['uncertain_or_unmapped_records']+=1
            self.uncertain[cell,self.compatible(self.taxa.get(code))]=True

    def finish(self):
        known=self.eligible & (~self.uncertain|self.positive)
        truth=np.where(known,self.positive.astype(np.float32),-1.)
        fraction=known.sum(1)/np.maximum(self.eligible.sum(1),1)
        return truth,known,fraction,dict(self.counts)


def score_fixed_sets(truth,known,rank,count):
    """Ground-truth masks affect evaluation only, never sets or confidence."""
    y=np.asarray(truth);known=np.asarray(known,bool);rank=np.asarray(rank,int);count=np.asarray(count,int)
    if y.shape!=known.shape or len(rank)!=len(y) or len(count)!=len(y):raise ValueError('Scoring identity mismatch')
    if np.any((y>0)&~known) or np.any(y[known]<0):raise ValueError('Unknown/positive mismatch')
    chosen=np.arange(rank.shape[1])[None,:]<count[:,None]
    observable=chosen & np.take_along_axis(known,rank,axis=1)
    hits=(observable & np.take_along_axis(y>0,rank,axis=1)).sum(1)
    n_truth=(y>0).sum(1);n_prediction=observable.sum(1);denominator=n_truth+n_prediction
    zero=2*hits/np.maximum(denominator,1)
    return {'F1':np.where(denominator==0,1.,zero),'F1_zero_empty':zero,'hits':hits,
            'known_truth_count':n_truth,'known_prediction_count':n_prediction,
            'unknown_prediction_count':count-n_prediction}


def site_primary_comparison(quality,predicted,mean_quality,candidate,baseline,sites,repetitions=4000,confidence=.975,seed=20261116):
    quality=np.asarray(quality,float);predicted=np.asarray(predicted,float)
    a,b=np.asarray(candidate,bool),np.asarray(baseline,bool)
    if any(v.shape!=quality.shape for v in [predicted,a,b]):raise ValueError('Paired comparison shape mismatch')
    names,inverse=np.unique(np.asarray(sites).astype(str),return_inverse=True)
    gain=(mean_quality-quality)**2-(predicted-quality)**2
    columns=[np.ones(len(quality)),gain,a,b,quality*a,quality*b]
    sums=np.array([np.bincount(inverse,weights=v,minlength=len(names)) for v in columns])
    draws=np.random.default_rng(seed).integers(len(names),size=(repetitions,len(names)))
    boot=sums[:,draws].sum(2);valid=(boot[0]>0)&(boot[2]>0)&(boot[3]>0)
    mse=boot[1,valid]/boot[0,valid];f1=boot[4,valid]/boot[2,valid]-boot[5,valid]/boot[3,valid]
    tail=(1-confidence)/2
    return {'MSE_gain_mean_minus_HGB':float(gain.mean()),'HGB_minus_SDC_selective_F1':float(quality[a].mean()-quality[b].mean()),
            'MSE_gain_interval':np.quantile(mse,[tail,1-tail]).tolist(),'F1_difference_interval':np.quantile(f1,[tail,1-tail]).tolist(),
            'sites':len(names),'cells':len(quality),'valid_draws':int(valid.sum()),'confidence':confidence}
