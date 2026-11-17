"""Selected-key access and chronological, non-favorable morphospecies joins."""
from datetime import datetime,timezone


def csv_byte_records(payload):
    data=payload if payload.endswith(b'\n') else payload+b'\n'
    fields=[];value=bytearray();position=0;quoted=False;start=True
    while position<len(data):
        c=data[position]
        if c==34:
            if quoted and position+1<len(data) and data[position+1]==34:
                value.append(34);position+=2;continue
            if quoted:quoted=False
            elif start:quoted=True
            else:raise ValueError('Unexpected CSV quote')
            start=False
        elif not quoted and c in (44,10,13):
            fields.append(bytes(value));value.clear();start=True
            if c in (10,13):
                if c==13 and position+1<len(data) and data[position+1]==10:position+=1
                yield fields;fields=[]
        else:value.append(c);start=False
        position+=1
    if quoted:raise ValueError('Unclosed CSV quote')


def selected_morpho_rows(payload,selected_keys):
    records=iter(csv_byte_records(payload));header=[v.decode('utf-8-sig') for v in next(records)]
    if len(set(header))!=len(header):raise ValueError('Duplicate morphospecies columns')
    site,morpho=header.index('siteID'),header.index('morphospeciesID')
    allowed={'siteID','morphospeciesID','morphospeciesResolvedDate','taxonID','identificationQualifier','dataQF'}
    for record in records:
        if len(record)!=len(header):raise ValueError('Morphospecies row width mismatch')
        key=(record[site].decode('utf-8'),record[morpho].decode('utf-8'))
        if key in selected_keys:
            yield {name:record[i].decode('utf-8') for i,name in enumerate(header) if name in allowed}


def latest_resolution(rows):
    dated=[]
    for row in rows:
        value=row.get('morphospeciesResolvedDate','').strip()
        if not value:continue
        try:
            date=datetime.fromisoformat(value.replace('Z','+00:00'))
            if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
        except ValueError:continue
        dated.append((date,row))
    if not dated:return None,'no_dated_resolution'
    date=max(d for d,_ in dated);latest=[r for d,r in dated if d==date]
    concepts={(r.get('taxonID',''),r.get('identificationQualifier',''),r.get('dataQF','')) for r in latest}
    if len(concepts)!=1:return None,'conflicting_latest_resolution'
    chosen=latest[0]
    if not chosen.get('taxonID','').strip():return None,'latest_resolution_missing_taxon'
    return chosen,'dated_resolution'
