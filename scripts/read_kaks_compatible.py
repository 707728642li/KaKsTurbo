"""Standalone, streaming KaKs table reader; retains literal NA sequence IDs."""
import csv
import math
NUMERIC = {'Ka', 'Ks', 'Ka/Ks', 'P-Value(Fisher)', 'Length', 'S-Sites', 'N-Sites',
           'Substitutions', 'Syn-Subs', 'Nonsyn-Subs', 'Divergence-Distance', 'ML-Score', 'AICc', 'Akaike-Weight'}

def read_kaks_compatible(path):
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        if len(reader.fieldnames or []) != 22 or reader.fieldnames[:6] != ['Sequence','Method','Ka','Ks','Ka/Ks','P-Value(Fisher)']:
            raise ValueError('Expected a 22-column KaKs_Calculator table')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed table row')
            for name in NUMERIC:
                row[name] = None if row[name] == 'NA' else float(row[name])
                if row[name] is not None and not math.isfinite(row[name]):
                    raise ValueError('Nonfinite table value')
            yield row
