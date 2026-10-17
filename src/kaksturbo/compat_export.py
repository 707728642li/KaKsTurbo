"""Read-only legacy-layout export. No model fitting or statistic estimation."""
from collections import Counter
import csv
import json
import os
from pathlib import Path
from ._paths import absolute_path
import tempfile

from .checkpoint import atomic_json, file_sha256, load_json, utc_now, verify_hashes
from .errors import ConfigurationError, IntegrityError
from .models import MODEL_ORDER, VERSION
from .results import CORE_FIELDS, parse_row

# Exact header/order from KaKs_Calculator 3.0 src/KaKs.cpp (22 actual columns).
LEGACY_FIELDS = (
    'Sequence', 'Method', 'Ka', 'Ks', 'Ka/Ks', 'P-Value(Fisher)', 'Length',
    'S-Sites', 'N-Sites', 'Fold-Sites(0:2:4)', 'Substitutions', 'Syn-Subs',
    'Nonsyn-Subs', 'Fold-Syn-Subs(0:2:4)', 'Fold-Nonsyn-Subs(0:2:4)',
    'Divergence-Distance', 'Substitution-Rate-Ratio(rTC:rAG:rTA:rCG:rTG:rCA/rCA)',
    'GC(1:2:3)', 'ML-Score', 'AICc', 'Akaike-Weight', 'Model',
)
# Historical native reproduction header; source Base::parseOutput uses the
# same ordered Sd/Nd/Si/Vi/t fields, while 3.0 KaKs.cpp renames their labels.
REPRODUCTION_ALIASES = {
    'Syn-Subs': 'S-Substitutions', 'Nonsyn-Subs': 'N-Substitutions',
    'Fold-Syn-Subs(0:2:4)': 'Fold-S-Substitutions(0:2:4)',
    'Fold-Nonsyn-Subs(0:2:4)': 'Fold-N-Substitutions(0:2:4)',
    'Divergence-Distance': 'Divergence-Time',
}
REPRODUCTION_FIELDS = tuple(REPRODUCTION_ALIASES.get(k,k) for k in LEGACY_FIELDS)
CORE_MAP = {'Sequence': 'pair_id', 'Method': 'method', 'Ka': 'Ka', 'Ks': 'Ks',
            'Ka/Ks': 'Ka_Ks', 'Substitutions': 'observed_SNP', 'ML-Score': 'lnL'}
ROBUST_OMISSIONS = {
    'P-Value(Fisher)': 'not_estimated: robust MA has no validated Fisher significance test',
    'Fold-Sites(0:2:4)': 'not_exported: no validated legacy fold-site statistic',
    'Syn-Subs': 'semantic_difference: expected S events are not relabelled as legacy Syn-Subs',
    'Nonsyn-Subs': 'semantic_difference: expected N events are not relabelled as legacy Nonsyn-Subs',
    'Fold-Syn-Subs(0:2:4)': 'not_exported: no validated legacy fold-event statistic',
    'Fold-Nonsyn-Subs(0:2:4)': 'not_exported: no validated legacy fold-event statistic',
    'Divergence-Distance': 'semantic_difference: codon-path/divergence time is not relabelled as legacy distance',
    'Substitution-Rate-Ratio(rTC:rAG:rTA:rCG:rTG:rCA/rCA)': 'not_exported: no validated legacy formatted rate-ratio mapping',
    'GC(1:2:3)': 'not_exported: legacy GC definition and case convention are not reconstructed',
    'AICc': 'not_applicable: candidate conditional AICc is not an MA-level legacy AICc',
    'Akaike-Weight': 'not_applicable: a model average has no single candidate Akaike weight',
    'Model': 'not_applicable: best candidate is not the identity of the complete model average',
}
LIMITATIONS = [
    'Table-layout compatibility does not imply estimator equivalence or numerical equality to KaKs_Calculator 3.0.',
    'Method MA denotes the requested estimator family; source mode/backend/version are recorded here, not disguised as the baseline tool.',
    'In robust mode ML-Score is the saved robust MA lnL, not the historical objective; S/N sites likewise retain robust definitions.',
    'Finite results do not certify convergence or positive selection. Retain source quality_summary and model diagnostics.',
    'Readers requiring finite Fisher P values cannot perform significance analysis on robust exports.',
    'Some third-party readers treat a literal Sequence ID NA as missing. IDs are never rewritten; use the supplied safe readers for such IDs.',
]


def _read_rows(path, fields=None):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if fields is not None and reader.fieldnames != list(fields):
            raise IntegrityError('Unexpected result header: ' + path.name)
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise IntegrityError('Malformed result row: ' + path.name)
            yield row


def _check_value(value):
    if not isinstance(value, str) or not value or any(c in value for c in '\t\r\n'):
        raise IntegrityError('Invalid unquoted TSV cell in compatibility export')
    return value


def export_kaks_calculator(directory, output_dir):
    """Verify saved final artifacts, then publish a separate new export directory.

    Works on completed public schema-2 runs, including older interface versions.
    Does not require .work, a native engine, the original input or a GPU.
    """
    source, target = absolute_path(directory), absolute_path(output_dir)
    if target.exists() or target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ConfigurationError('Export requires a new directory outside the original run directory')
    manifest_path = source / 'run_manifest.json'
    manifest_digest = file_sha256(manifest_path)
    manifest = load_json(manifest_path)
    if (manifest.get('schema') not in {'kaksturbo.run_manifest/2', 'kaksforge.run_manifest/2'}
            or manifest.get('complete') is not True or manifest.get('mode') not in {'robust', 'reproduce'}
            or manifest.get('method') != 'MA' or manifest.get('model_count') != 14
            or manifest.get('models') != list(MODEL_ORDER)):
        raise IntegrityError('Export requires a completed fourteen-model MA public result manifest')
    count, hashes = manifest.get('count'), manifest.get('artifacts_sha256')
    if type(count) is not int or count < 1 or not isinstance(hashes, dict):
        raise IntegrityError('Invalid completed result count or artifact hashes')
    required = {'result.tsv', 'native_result.tsv', 'results.jsonl'}
    if not required.issubset(hashes):
        raise IntegrityError('Required final tables/JSONL are not registered in the manifest')
    # Only consumed source files are required. Model diagnostics may have been
    # archived separately; this formatter does not certify or refit those files.
    verified_hashes = {k: hashes[k] for k in sorted(required)}
    verify_hashes(source, verified_hashes)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + target.name + '.export-', dir=target.parent))
    missing_counts = Counter()
    source_mode = manifest['mode']
    total = 0
    try:
        with (source / 'native_result.tsv').open(encoding='utf-8-sig', newline='') as f:
            native_header = next(csv.reader(f,delimiter='\t'), [])
        if source_mode == 'reproduce' and tuple(native_header) not in (LEGACY_FIELDS,REPRODUCTION_FIELDS):
            raise IntegrityError('Unexpected native reproduction header')
        aliases = REPRODUCTION_ALIASES if tuple(native_header) == REPRODUCTION_FIELDS else {}
        native_rows = _read_rows(source / 'native_result.tsv')
        with (stage / 'result.kaks').open('w', encoding='utf-8', newline='\n') as result, \
                (stage / 'missing_fields.tsv').open('w', encoding='utf-8', newline='') as reasons, \
                (source / 'results.jsonl').open(encoding='utf-8') as json_rows:
            result.write('\t'.join(LEGACY_FIELDS) + '\n')
            rw = csv.writer(reasons, delimiter='\t', lineterminator='\n')
            rw.writerow(['pair_index', 'pair_id', 'field', 'reason'])
            for index, raw in enumerate(_read_rows(source / 'result.tsv', CORE_FIELDS)):
                core = parse_row(raw)
                native = next(native_rows, None)
                try:
                    envelope = json.loads(next(json_rows))
                except (StopIteration, ValueError) as exc:
                    raise IntegrityError('Missing or malformed result JSONL row') from exc
                if (envelope.get('global_pair_index') != index or envelope.get('pair_id') != core['pair_id']
                        or envelope.get('record') != core or core['mode'] != source_mode
                        or core['method'] != 'MA' or native is None
                        or native.get('Sequence') != core['pair_id'] or native.get('Method') != 'MA'):
                    raise IntegrityError('Core/native/JSONL result identity, mode or row order differs')
                for old, new in [('Ka', 'Ka'), ('Ks', 'Ks'), ('Ka/Ks', 'Ka_Ks')]:
                    n = native.get(old)
                    if n is None or (None if n == 'NA' else float(n)) != core[new]:
                        raise IntegrityError('Core and native scientific values differ: ' + old)
                if source_mode == 'reproduce':
                    row = {k: _check_value(native[aliases.get(k,k)]) for k in LEGACY_FIELDS}
                    why = {k: 'historical_reported_NA: preserved from native reproduction output'
                           for k in LEGACY_FIELDS[2:] if row[k] == 'NA'}
                else:
                    row = dict.fromkeys(LEGACY_FIELDS, 'NA')
                    row.update({old: raw[new] for old, new in CORE_MAP.items()})
                    row['Length'] = 'NA' if core['eligible_codons'] is None else str(3 * core['eligible_codons'])
                    for old, new in [('S-Sites', 'S-sites'), ('N-Sites', 'N-sites')]:
                        if new not in native:
                            raise IntegrityError('Robust native result is missing ' + new)
                        row[old] = native[new]
                        if row[old] != 'NA':
                            import math
                            if not math.isfinite(float(row[old])) or float(row[old]) < 0:
                                raise IntegrityError('Invalid native site count')
                    why = dict(ROBUST_OMISSIONS)
                    for k in LEGACY_FIELDS[2:]:
                        if row[k] in ('NA', '') and k not in why:
                            row[k] = 'NA'
                            why[k] = 'source_value_unavailable: ' + (core['reason'] or core['status'])
                if index >= count:
                    raise IntegrityError('More results than the completed manifest count')
                result.write('\t'.join(_check_value(row[k]) for k in LEGACY_FIELDS) + '\n')
                for k, reason in why.items():
                    if row[k] != 'NA':
                        raise IntegrityError('Missing-value reason disagrees with output')
                    rw.writerow([index, core['pair_id'], k, reason])
                    missing_counts[k] += 1
                total += 1
            if total != count or next(native_rows, None) is not None or json_rows.read().strip():
                raise IntegrityError('Final table/JSONL counts differ from the manifest')
        # Detect modifications during reading; only publish after revalidation.
        verify_hashes(source, verified_hashes)
        if file_sha256(manifest_path) != manifest_digest:
            raise IntegrityError('Source manifest changed during export')
        info = dict(schema='kaksturbo.compatibility_export/1', complete=True, exporter_version=VERSION,
            created_at=utc_now(), format='KaKs_Calculator 3.0 22-column TSV layout', count=total,
            source_directory=str(source), source_version=manifest.get('version'), source_mode=source_mode,
            source_backends=manifest.get('backends_used'), source_manifest_sha256=manifest_digest,
            source_artifacts_sha256=hashes, columns=list(LEGACY_FIELDS), missing_counts=dict(missing_counts),
            verified_source_artifacts_sha256=verified_hashes,
            unverified_nonconsumed_artifacts=sorted(set(hashes)-required),
            native_source_columns=native_header,
            mapping=({'policy': 'verbatim native reproduction cells; documented header aliases only',
                      'output_to_native_header_aliases':aliases} if source_mode == 'reproduce' else
                {'core_columns': CORE_MAP, 'Length': '3 * eligible_codons; cleaned length in nucleotides',
                 'S-Sites': 'native S-sites; robust estimator', 'N-Sites': 'native N-sites; robust estimator',
                 'NA_policy': ROBUST_OMISSIONS}), limitations=LIMITATIONS,
            fitting_performed=False, scientific_values_reestimated=False, input_files_modified=False)
        (stage / 'README.txt').write_text(
            'KaKsTurbo optional KaKs_Calculator table-layout export\n'
            'result.kaks: exact 22-column header/order, tab-delimited, no comment preamble.\n'
            'missing_fields.tsv: a reason for every numeric/statistical NA cell, indexed by source pair order.\n'
            'compatibility.json: original version/mode/backend, source hashes, field mapping and limitations.\n'
            'This table does not replace result.tsv or the full model/quality diagnostics.\n'
            'Only the three consumed result tables/JSONL and manifest are verified; model/QC files are not recertified.\n'
            'Use a tab-delimited reader with quote/comment processing disabled; preserve Sequence as text.\n'
            + '\n'.join(LIMITATIONS) + '\n', encoding='utf-8')
        info['artifacts_sha256'] = {p.name: file_sha256(p) for p in stage.iterdir() if p.is_file()}
        atomic_json(stage / 'compatibility.json', info)
        if target.exists():
            raise ConfigurationError('Export destination appeared during writing; refusing to overwrite')
        os.rename(stage, target)
        return dict(schema='kaksturbo.export_command/1', complete=True, directory=str(target),
                    result=str(target / 'result.kaks'), count=total, source_mode=source_mode,
                    missing_cells=sum(missing_counts.values()), fitting_performed=False, version=VERSION)
    except Exception:
        # Keep a failed private staging directory for diagnosis; never publish it as complete.
        if stage.is_dir():
            atomic_json(stage / 'export_failure.json', {'complete': False, 'source': str(source), 'target': str(target)})
        raise
