"""Allowlisted CSV-byte projection; discarded outcome values are never decoded."""
import csv
import io


def project_csv_bytes(payload, allowed_fields):
    """Return unique metadata tuples, without retaining outcome-field multiplicity.

    CSV quoting is parsed at byte level. Only column names and allowed field
    values undergo UTF-8 decoding. Raw observation files must not be persisted.
    """
    allowed = set(allowed_fields)
    header = None
    capture_indices = None
    fields, value = {}, bytearray()
    column, position, quoted, field_start = 0, 0, False, True
    rows = set()
    # A terminal newline simplifies final-record handling without changing fields.
    data = payload if payload.endswith(b'\n') else payload + b'\n'
    while position < len(data):
        char = data[position]
        capture = header is None or column in capture_indices
        if char == 34:
            if quoted:
                if position + 1 < len(data) and data[position + 1] == 34:
                    if capture:
                        value.append(34)
                    position += 2
                    continue
                quoted = False
            elif field_start:
                quoted = True
            else:
                raise ValueError('Unexpected quote in unquoted CSV field')
            field_start = False
        elif not quoted and char in (44, 10, 13):
            if capture:
                fields[column] = bytes(value)
            value.clear()
            column += 1
            field_start = True
            if char in (10, 13):
                if char == 13 and position + 1 < len(data) and data[position + 1] == 10:
                    position += 1
                if header is None:
                    header = [fields[i].decode('utf-8-sig') for i in range(column)]
                    if len(set(header)) != len(header):
                        raise ValueError('Duplicate CSV field names')
                    capture_indices = [i for i, name in enumerate(header) if name in allowed]
                    if not capture_indices:
                        raise ValueError('No allowed metadata fields found')
                else:
                    if column != len(header):
                        raise ValueError('CSV row width differs from header')
                    rows.add(tuple(fields[i].decode('utf-8') for i in capture_indices))
                fields, column = {}, 0
        else:
            if capture:
                value.append(char)
            field_start = False
        position += 1
    if quoted:
        raise ValueError('Unclosed quoted CSV field')
    if header is None:
        raise ValueError('CSV header is absent')
    return [header[i] for i in capture_indices], sorted(rows)


def metadata_csv_bytes(columns, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')
