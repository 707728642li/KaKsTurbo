"""Explicit conditional true-pi injection for the registered fixed-factor Monte Carlo data.

The only mutation is prepared.values[i][2:66]. Native adapter pair contexts,
input sequences, GC, masks and patterns are never modified. Validation needs
an explicit boolean AND the caller's separate frozen-configuration gate.
This module cannot authorize or execute a fit.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path

BASES = 'TCAG'
CODE1 = 'FFLLSSSSYY!!CC!WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
CODONS64 = [''.join(c) for c in itertools.product(BASES, repeat=3)]
SENSE64 = [i for i, aa in enumerate(CODE1) if aa != '!']
SENSE = [CODONS64[i] for i in SENSE64]
INDEX = {c: i for i, c in enumerate(SENSE)}
MODEL_CONTRACT = 'pure-rates-strict-positive-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError('Conditional truth-frequency injection: ' + message)


def parse_registered_axt(data):
    """Read the MC generator's exact canonical LF AXT, preserving bytes."""
    require(isinstance(data, bytes) and b'\r' not in data, 'canonical LF AXT bytes required')
    result = {}
    for block in data.split(b'\n\n'):
        if not block:
            continue
        lines = block.split(b'\n')
        require(len(lines) == 3, 'each MC AXT block needs exactly three lines')
        case_id, seq0, seq1 = [line.decode('ascii') for line in lines]
        require(case_id and case_id not in result, 'duplicate or empty AXT id')
        result[case_id] = (seq0, seq1, block + b'\n\n')
    require(result, 'empty AXT')
    return result


def input_statistics(seq0, seq1):
    require(len(seq0) == len(seq1) and len(seq0) > 0 and len(seq0) % 3 == 0, 'bad paired codon dimensions')
    a = [seq0[i:i+3] for i in range(0, len(seq0), 3)]
    b = [seq1[i:i+3] for i in range(0, len(seq1), 3)]
    require(all(c in INDEX for c in a + b), 'only unmasked registered code1 sense codons are supported')
    patterns = Counter((max(INDEX[x], INDEX[y]), min(INDEX[x], INDEX[y])) for x, y in zip(a, b))
    observed = sum(x != y for x, y in zip(seq0, seq1))
    joined = seq0 + seq1
    gc = [sum(c in 'GC' for c in joined) / len(joined)]
    gc.extend(sum(c in 'GC' for c in joined[p::3]) / (len(joined) / 3) for p in range(3))
    return len(seq0), observed, [[i, j, n] for (i, j), n in sorted(patterns.items())], gc


def load_registered_truth(truth_jsonl, scenarios_json):
    """Validate all 96 scenarios and every registered replication before any mutation."""
    truth_path, scenario_path = Path(truth_jsonl).resolve(), Path(scenarios_json).resolve()
    require(truth_path.parent == scenario_path.parent, 'truth and scenarios must belong to one dataset directory')
    directory = truth_path.parent
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest.get('schema') == 'kakscuda.monte_carlo_registration/1' and manifest.get('complete') is True and
            manifest.get('sources_unchanged') is True, 'incomplete or unknown simulation manifest')
    require(truth_path.name == 'truth.jsonl' and scenario_path.name == 'scenarios.json', 'registered canonical filenames required')
    replicas = manifest.get('replicates_per_scenario')
    require(type(replicas) is int and 2 <= replicas <= 400 and replicas % 2 == 0,
            'registration requires an even replication count 2..400')
    per_split = replicas // 2
    total_count = 96 * replicas
    expected_splits = {'development': 96 * per_split, 'validation': 96 * per_split}
    require(manifest.get('pair_count') == total_count and manifest.get('scenario_count') == 96 and
            manifest.get('split_counts') == expected_splits and
            manifest.get('split_replicates_per_scenario') == per_split,
            'all registered scenario and split denominators must agree')
    require(not any(manifest.get(k, True) for k in ('fitter_invoked', 'GPU_used', 'native_backend_invoked', 'matrix_exponential_invoked')),
            'manifest is not a generation-only MC dataset')
    required_files = {'development.axt', 'validation.axt', 'truth.jsonl', 'scenarios.json'}
    require(set(manifest.get('input_sha256', {})) == required_files, 'unexpected or incomplete registered file list')
    for name, expected in manifest['input_sha256'].items():
        require(sha(directory / name) == expected, 'registered file SHA mismatch: ' + name)
    require(sha(directory / 'generate_publication_monte_carlo.py') == manifest['generator_sha256'], 'archived generator SHA mismatch')
    for key in ('generator', 'random.py', '_random_binary'):
        item = manifest['source_change_audit'][key]
        require(item['before'] == item['after'], 'generation source changed: ' + key)
    prior_info = manifest.get('prior_registration')
    require(isinstance(prior_info, dict), 'prior design registration missing')
    prior_path = Path(prior_info['path'])
    require(sha(prior_path) == prior_info['sha256'], 'prior registration manifest SHA mismatch')
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    require(prior.get('schema') == 'kakscuda.monte_carlo_design/1' and prior.get('complete') is True and
            prior['master_seed'] == manifest['master_seed'] and prior['generator_sha256'] == manifest['generator_sha256'] and
            prior['replicates_per_scenario'] == replicas and prior.get('scenario_count') == 96 and
            prior.get('planned_pair_count') == total_count and prior.get('planned_split_counts') == expected_splits,
            'prior design settings or coverage differ')
    require(sha(prior_path.parent / 'case_design.json') == prior['case_design_sha256'] and
            sha(prior_path.parent / 'scenarios.json') == prior['scenarios_sha256'] == sha(scenario_path), 'prior case/scenario SHA mismatch')
    require(sha(prior_path.parent / 'generate_publication_monte_carlo.py') == manifest['generator_sha256'], 'prior generator SHA mismatch')
    scenarios = json.loads(scenario_path.read_text(encoding='utf-8'))
    plans = json.loads((prior_path.parent / 'case_design.json').read_text(encoding='utf-8'))
    truth = [json.loads(line) for line in truth_path.read_text(encoding='utf-8').splitlines()]
    require(len(scenarios) == 96 and len(truth) == len(plans) == total_count, 'complete design/truth coverage required')
    case_ids, seed_ids, expected_scenarios = set(), set(), set()
    for i, scenario in enumerate(scenarios):
        copy = dict(scenario); expected = copy.pop('scenario_sha256')
        require(digest(copy) == expected and scenario['index'] == i, 'scenario content/index SHA mismatch')
        require(scenario['genetic_code'] == 1 and scenario['support_policy'] == 'StrictPositive' and
                scenario['model_contract'] == MODEL_CONTRACT, 'wrong genetic-code/support/model semantics')
        require(scenario['generating_model'] in ('HKY', 'GTR') and
                scenario['model_id'] == {'HKY': 3, 'GTR': 13}[scenario['generating_model']], 'wrong generating model')
        pi = [float.fromhex(p) for p in scenario['pi61_hex']]
        require(len(pi) == 61 and all(math.isfinite(p) and 0 < p <= 1 for p in pi) and
                abs(math.fsum(pi) - 1.) <= 1e-12, 'invalid strictly positive normalized true pi')
        expected_scenarios.add((scenario['generating_model'], scenario['frequency_profile'], scenario['codon_count'], scenario['omega'], scenario['time']))
    require(expected_scenarios == set(itertools.product(('HKY', 'GTR'), ('balanced', 'position_GC_skew'),
                                                     (150, 600, 2400), (.1, .5, 1., 2.), (.03, .3))), 'registered factorial coverage differs')
    all_axt = {}
    split_counts, cells = Counter(), Counter()
    for split in ('development', 'validation'):
        for case_id, sequences in parse_registered_axt((directory / (split + '.axt')).read_bytes()).items():
            require(case_id not in all_axt, 'duplicate id across registered splits')
            all_axt[case_id] = (split, *sequences)
    require(len(all_axt) == total_count, 'every unique registered AXT pair required')
    by_id = {}
    for row, plan in zip(truth, plans):
        case_id = row['id']; require(case_id not in case_ids and row['seed_sha256'] not in seed_ids, 'duplicate id or seed in truth')
        case_ids.add(case_id); seed_ids.add(row['seed_sha256'])
        require(all(row.get(k) == v for k, v in plan.items()), 'truth row differs from prior seed/case plan')
        require(row['schema'] == 'kakscuda.monte_carlo_pair/1' and 0 <= row['scenario_index'] < 96, 'invalid row schema/scenario index')
        scenario = scenarios[row['scenario_index']]
        require(row['scenario_sha256'] == scenario['scenario_sha256'], 'row/scenario mismatch')
        key = f'kakscuda.monte_carlo_pair/1|{manifest["master_seed"]}|{row["split"]}|{scenario["id"]}|rep{row["replicate_within_split"]}'
        seed = hashlib.sha256(key.encode()).hexdigest()
        require(seed == row['seed_sha256'] and hex(int(seed, 16)) == row['seed_integer_hex'], 'seed derivation mismatch')
        plain = dict(row); expected = plain.pop('record_sha256')
        require(digest(plain) == expected, 'truth row content SHA mismatch')
        require(case_id in all_axt, 'truth id missing from registered inputs')
        split, seq0, seq1, block = all_axt[case_id]
        require(split == row['split'] and split in ('development', 'validation'), 'input/truth split mismatch')
        require(hashlib.sha256(block).hexdigest() == row['pair_AXT_sha256'] and
                hashlib.sha256(seq0.encode()).hexdigest() == row['sequence0_sha256'] and
                hashlib.sha256(seq1.encode()).hexdigest() == row['sequence1_sha256'], 'input/record/sequence hash mismatch')
        nt, snp, patterns, gc = input_statistics(seq0, seq1)
        require(nt == row['cleaned_nt_length'] == 3 * row['codon_count'] == 3 * scenario['codon_count'] and
                snp == row['observed_SNP'] and patterns == row['triangular_patterns'] and row['masked_codons'] == 0,
                'input length/SNP/pattern/mask differs from truth')
        require(type(row['replicate_within_split']) is int and 0 <= row['replicate_within_split'] < per_split and row['replicate_global'] ==
                row['replicate_within_split'] + (per_split if split == 'validation' else 0), 'replication/split identity mismatch')
        split_counts[split] += 1; cells[(row['scenario_index'], split)] += 1
        by_id[case_id] = {'record': row, 'scenario': scenario, 'sequences': (seq0, seq1),
                          'block': block, 'gc': gc}
    require(dict(split_counts) == manifest['split_counts'] and len(cells) == 192 and all(n == per_split for n in cells.values()),
            'split/cell coverage incomplete')
    return by_id, {'manifest': manifest, 'manifest_sha256': sha(manifest_path), 'truth_sha256': sha(truth_path),
                   'scenarios_sha256': sha(scenario_path), 'case_design_sha256': prior['case_design_sha256'],
                   'prior_manifest_sha256': prior_info['sha256'], 'generator_sha256': manifest['generator_sha256']}


def inject_truth_frequencies(prepared, truth_jsonl, scenarios_json, allow_validation=False):
    """Atomically validate then inject true pi into an existing prepared subset.

    Validation injection requires ``allow_validation is True``; the fitting
    driver must independently verify its frozen source/configuration authority
    before using that option. This boolean is not an optimization authorization.
    Equal-frequency candidate models retain the core's declared uniform pi;
    injection supplies the fixed conditional pi for unequal-frequency models.
    """
    require(type(allow_validation) is bool, 'allow_validation must be an explicit boolean')
    by_id, provenance = load_registered_truth(truth_jsonl, scenarios_json)
    ids = list(prepared.ids)
    require(ids and len(set(ids)) == len(ids), 'prepared ids must be unique and nonempty')
    for name in ('meta', 'values', 'z0', 'z1', 'weights', 'gc'):
        require(len(getattr(prepared, name)) == len(ids), 'prepared array count mismatch: ' + name)
    require(len({id(row) for row in prepared.values}) == len(ids), 'prepared frequency rows must not alias')
    raw = parse_registered_axt(prepared.raw_input)
    require(hashlib.sha256(prepared.raw_input).hexdigest() == prepared.input_sha256, 'prepared raw-input SHA differs')
    # The raw input can be a full registered split or a selected/reordered exact
    # subset. Every block present must be registered and unchanged.
    for case_id, (seq0, seq1, block) in raw.items():
        require(case_id in by_id and block == by_id[case_id]['block'], 'raw input contains an unregistered or changed record')
    updates, pair_audit = [], []
    split_counts = Counter()
    preserved = {'meta': prepared.meta, 'z0': prepared.z0, 'z1': prepared.z1,
                 'weights': prepared.weights, 'gc': prepared.gc, 'value_prefix': [r[:2] for r in prepared.values]}
    preserved_sha = digest(preserved)
    for i, case_id in enumerate(ids):
        require(case_id in by_id and case_id in raw, 'prepared id not present in the registered raw input')
        item = by_id[case_id]; row, scenario = item['record'], item['scenario']
        require(row['split'] != 'validation' or allow_validation is True,
                'validation injection needs explicit allow_validation=True and caller frozen-configuration authorization')
        require(isinstance(prepared.values[i], list) and len(prepared.values[i]) == 66, 'mutable prepared 66-double list required')
        require(prepared.values[i][:2] == [row['cleaned_nt_length'], row['observed_SNP']], 'prepared cleaned length/SNP mismatch')
        patterns = row['triangular_patterns']
        require(len(prepared.meta[i]) == 4 and prepared.meta[i][0] >= 0 and
                prepared.meta[i][1:] == [len(patterns), row['codon_count'], 0], 'prepared pattern metadata mismatch')
        require(len(prepared.z0[i]) == len(prepared.z1[i]) == len(prepared.weights[i]) == len(patterns),
                'prepared pattern-array lengths differ')
        require(list(zip(prepared.z0[i], prepared.z1[i], prepared.weights[i])) == [tuple(p) for p in patterns],
                'prepared patterns or weights differ')
        require(len(prepared.gc[i]) == 4 and all(math.isfinite(g) and abs(g - ref) <= 1e-14 for g, ref in zip(prepared.gc[i], item['gc'])),
                'prepared pre-filter GC differs from the unchanged input')
        pi61 = [float.fromhex(p) for p in scenario['pi61_hex']]
        pi64 = pi61 + [0., 0., 0.]
        # Full raw TCAG64 representation is audit metadata only. The native
        # prepared ABI is compressed sense61 followed by three unused zeros.
        full64 = [0.] * 64
        for index, value in zip(SENSE64, pi61): full64[index] = value
        original = prepared.values[i][2:66]
        require(len(original) == 64 and all(math.isfinite(p) and p >= 0 for p in original) and
                abs(math.fsum(original[:61]) - 1.) <= 1e-12 and original[61:] == [0., 0., 0.], 'invalid original prepared pi ABI')
        updates.append(pi64)
        split_counts[row['split']] += 1
        pair_audit.append({'pair': case_id, 'scenario_index': row['scenario_index'], 'split': row['split'],
                           'record_sha256': row['record_sha256'], 'pair_AXT_sha256': row['pair_AXT_sha256'],
                           'original_prepared_pi64_hex': [p.hex() for p in original],
                           'injected_compressed_pi64_hex': [p.hex() for p in pi64],
                           'true_full_TCAG64_with_stop_zeros_hex': [p.hex() for p in full64],
                           'generating_model': scenario['generating_model']})
    # No writes occur before every selected input has passed all gates.
    old_rows = [r[2:66] for r in prepared.values]
    try:
        for row, pi64 in zip(prepared.values, updates): row[2:66] = pi64
        require(digest(preserved) == preserved_sha, 'a preserved field unexpectedly changed')
    except BaseException:
        for row, old in zip(prepared.values, old_rows): row[2:66] = old
        raise
    return {'schema': 'kakscuda.conditional_truth_frequency_injection/2', 'complete': True,
            'conditioned_on_true_stationary_pi': True, 'frequency_estimated': False,
            'model_contract': MODEL_CONTRACT, 'genetic_code': 1, 'support_policy': 'StrictPositive',
            'prepared_pi_layout': 'TCAG sense61 compressed, then three unused zeros; not raw TCAG64 positions',
            'equal_frequency_candidate_policy': 'Declared even model indices retain uniform sense61 frequencies',
            'pair_count': len(ids), 'split_counts': dict(split_counts), 'validated_truth_pair_count': provenance['manifest']['pair_count'],
            'registered_replications_per_scenario': provenance['manifest']['replicates_per_scenario'],
            'registered_split_denominators': provenance['manifest']['split_counts'],
            'prepared_raw_input_sha256': prepared.input_sha256, 'dataset_provenance': {k: v for k, v in provenance.items() if k != 'manifest'},
            'injection_source_sha256': sha(__file__), 'preserved_nonfrequency_fields_sha256': preserved_sha,
            'native_pair_contexts_modified': False, 'mutation_scope': 'prepared.values[i][2:66] only',
            'validation_option_explicitly_enabled': allow_validation,
            'outer_frozen_source_configuration_verification_required': bool(split_counts['validation']),
            'fitting_authorized_by_this_function': False, 'fits_performed': False, 'GPU_used': False,
            'pairs': pair_audit}
