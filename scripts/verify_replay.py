#!/usr/bin/env python3
"""Independently audit raw PPL/MK reports and compare a fresh v1 GPU replay.

Only the Python standard library is required. Exit 1 means an identity,
structure, or arithmetic check failed. Exit 0 means both reports are internally
valid; inspect ``exact_replay`` and ``status`` for numerical/output differences.
No tokenizer decoding or model inference is performed by this offline audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'source_checkpoint_sha256': '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb',
    'tokenizer_sha256': '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09',
    'protocol_sha256': 'dee9457bfaca067b63837dcd3bc94bf9ef9a3d805b1ab04227db05a4b92c54c2',
    'adapter_sha256': 'e8b2b4dfe69f8e85dc9e147c9aeaa3297cff14f4043e558bb1795ad476c1fca0',
}
PUBLISHED_METADATA_PINS = {
    'data_manifest_sha256': '227737dd5fbb301d76e583457ff8ee29246047661b3e9a64ebd068a6f34f376c',
    'dataset_fingerprint': '51d84d52d095f90d',
}
DATASET_PINS = {
    'dataset': 'Salesforce/wikitext', 'configuration': 'wikitext-2-raw-v1',
    'split': 'validation', 'revision_argument': 'b08601e04326c79dfdd32d625aee71d232d685c3',
    'document_join': 'two newline characters',
    'text_sha256': 'b44fb967f92b525731216a507d1fbc44b97446b7ef0a2b4d1a893f81f1bbc29e',
    'token_stream_sha256_int64le': '5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027',
    'total_tokens': 264765, 'tokenizer_sha256': PINS['tokenizer_sha256'],
    'automatic_special_tokens': False,
}
MATCH = re.compile(r'(?<!\d)\d{6}(?!\d)')
SHA = re.compile(r'[0-9a-f]{64}')
IDENTITY_FIELDS = ('id', 'condition', 'N', 'template', 'expected',
                   'prompt_token_sha256_int64le', 'prompt_tokens', 'cache_bytes')
PPL_IDENTITY_FIELDS = ('start', 'input_tokens', 'target_tokens', 'token_sha256_int64le')
ARITHMETIC_REL_TOL = 1e-12
ARITHMETIC_ABS_TOL = 1e-9


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)

    return json.loads(Path(path).read_text(), object_pairs_hook=reject_duplicates,
                      parse_constant=reject_constant)


def write_new_audit(path, result):
    """Preserve existing receipts, including if a path appears during auditing."""
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(serialized)


def expected_mk_order():
    return [(f'resurface-confirm-n{size}-t{template}-s{sample}' +
             ('-removed' if condition == 'target_removed' else ''),
             condition, size, template)
            for size in (16, 64) for template in range(3) for sample in range(64)
            for condition in ('normal', 'target_removed')]


class Audit:
    def __init__(self):
        self.errors = []
        self.decoded_outputs = {}

    def check(self, condition, message):
        if not condition:
            self.errors.append(message)

    def number(self, value, label, minimum=0):
        valid = type(value) in (int, float) and math.isfinite(value) and value >= minimum
        self.check(valid, label + ': expected finite number >= ' + str(minimum))
        if not valid:
            raise ValueError(label + ': invalid numeric value')
        return value

    def integer(self, value, label, minimum=0):
        self.check(type(value) is int and value >= minimum, label + ': invalid integer')

    def equal(self, actual, expected, label):
        self.check(type(actual) is type(expected) and actual == expected,
                   label + ': inconsistent value')

    def arithmetic(self, actual, expected, label):
        actual = self.number(actual, label, minimum=-math.inf)
        self.check(math.isclose(actual, expected, rel_tol=ARITHMETIC_REL_TOL,
                                abs_tol=ARITHMETIC_ABS_TOL),
                   f'{label}: stored={actual!r}, recomputed={expected!r}')

    def ppl(self, block, label):
        rows = block['windows']
        self.equal(len(rows), 130, label + '.window_count')
        self.equal(block['execution'], 'prefill', label + '.execution')
        self.equal(block['cache_dtype'], 'SSD scan internal precision', label + '.cache_dtype')
        self.equal(block['cache_bytes'], None, label + '.cache_bytes')
        self.equal(block['logits_chunk_tokens'], 64, label + '.logits_chunk_tokens')
        self.equal(block['score'], 'next-token cross entropy; no BOS/EOS added; each window starts from zero state',
                   label + '.score')
        losses, targets = [], []
        for index, row in enumerate(rows):
            tag = f'{label}.windows[{index}]'
            target_count = 2048 if index < 129 else 572
            self.equal(row['start'], index * 2048, tag + '.start')
            self.equal(row['target_tokens'], target_count, tag + '.target_tokens')
            self.equal(row['input_tokens'], target_count, tag + '.input_tokens')
            self.check(isinstance(row['token_sha256_int64le'], str) and
                       SHA.fullmatch(row['token_sha256_int64le']) is not None, tag + '.token_hash')
            loss = self.number(row['nll'], tag + '.nll')
            losses.append(loss)
            targets.append(row['target_tokens'])
            self.arithmetic(row['ppl'], math.exp(loss / target_count), tag + '.ppl')
        target_total, nll = sum(targets), math.fsum(losses)
        self.equal(target_total, 264764, label + '.target_count')
        self.equal(block['target_tokens'], target_total, label + '.target_tokens')
        self.arithmetic(block['nll'], nll, label + '.nll')
        ppl = math.exp(nll / target_total)
        self.arithmetic(block['ppl'], ppl, label + '.ppl')
        return {'windows': len(rows), 'target_tokens': target_total, 'nll_recomputed': nll,
                'ppl_recomputed': ppl, 'stored_nll_minus_recomputed': block['nll'] - nll,
                'stored_ppl_minus_recomputed': block['ppl'] - ppl}

    def mk(self, block, label, probe=False):
        rows = block['rows']
        order = expected_mk_order()[:8] if probe else expected_mk_order()
        self.equal(len(rows), len(order), label + '.row_count')
        self.equal(block['generation'], 'greedy max12, full256K, fresh FP16 cache', label + '.generation')
        self.equal(block['matching'], 'first standalone six-digit integer', label + '.matching')
        self.equal(len(set(row['id'] for row in rows)), len(rows), label + '.unique_ids')
        self.equal(len(set(row['prompt_token_sha256_int64le'] for row in rows)), len(rows),
                   label + '.unique_prompt_hashes')
        correct = []
        for index, row in enumerate(rows):
            tag = f'{label}.rows[{index}]'
            if index < len(order):
                self.equal(tuple(row[key] for key in ('id', 'condition', 'N', 'template')),
                           order[index], tag + '.ordered_case_identity')
            self.check(isinstance(row['expected'], str) and
                       re.fullmatch(r'[0-9]{6}', row['expected']) is not None, tag + '.expected')
            self.check(isinstance(row['output'], str), tag + '.output must be text')
            self.check(isinstance(row['prompt_token_sha256_int64le'], str) and
                       SHA.fullmatch(row['prompt_token_sha256_int64le']) is not None, tag + '.prompt_hash')
            self.integer(row['prompt_tokens'], tag + '.prompt_tokens', minimum=1)
            self.equal(row['cache_bytes'], 122028032, tag + '.cache_bytes')
            generated = row['generated_ids']
            self.check(isinstance(generated, list) and 1 <= len(generated) <= 12,
                       tag + '.generated_ids count outside greedy max12')
            self.check(all(type(token) is int and 0 <= token < 256000 for token in generated),
                       tag + '.generated_ids outside vocabulary')
            if generated:
                self.check(3 not in generated[:-1], tag + '.tokens after EOS')
                self.check(len(generated) == 12 or generated[-1] == 3,
                           tag + '.short generation without EOS')
            token_key = tuple(generated)
            if token_key in self.decoded_outputs:
                self.equal(row['output'], self.decoded_outputs[token_key], tag + '.same_ids_different_output')
            else:
                self.decoded_outputs[token_key] = row['output']
            match = MATCH.search(row['output'])
            prediction = match.group(0) if match else None
            score = prediction == row['expected']
            self.equal(row['prediction'], prediction, tag + '.prediction regex rescore')
            self.equal(row['correct'], score, tag + '.correct regex rescore')
            correct.append(score)
            if index % 2:
                self.equal(row['expected'], rows[index - 1]['expected'], tag + '.paired_expected')
        summary = {}
        for condition in ('normal', 'target_removed'):
            selected = [i for i, row in enumerate(rows) if row['condition'] == condition]
            count, successes = len(selected), sum(correct[i] for i in selected)
            summary[condition] = {'correct': successes, 'count': count,
                                  'accuracy': successes / count if count else None}
            stored = block['summary'][condition]
            self.equal(stored['correct'], successes, label + '.' + condition + '.correct')
            self.equal(stored['count'], count, label + '.' + condition + '.count')
            if count:
                self.arithmetic(stored['accuracy'], successes / count, label + '.' + condition + '.accuracy')
        per_n = {}
        for size in (16, 64):
            selected = [i for i, row in enumerate(rows) if row['condition'] == 'normal' and row['N'] == size]
            if selected:
                count, successes = len(selected), sum(correct[i] for i in selected)
                per_n[str(size)] = {'correct': successes, 'count': count, 'accuracy': successes / count}
        if not probe:
            self.equal(sum(r['prompt_tokens'] for r in rows), 517120, label + '.total_prompt_tokens')
            self.equal(max(r['prompt_tokens'] for r in rows), 1236, label + '.max_prompt_tokens')
        return {'summary': summary, 'normal_by_N': per_n,
                'generated_length_histogram': dict(sorted(Counter(len(r['generated_ids']) for r in rows).items()))}

    def paired_mk(self, report, label):
        before, after = report['baseline_mk']['rows'], report['adapter_mk']['rows']
        self.equal(len(before), len(after), label + '.paired row count')
        for index, (old, new) in enumerate(zip(before, after)):
            self.equal(tuple(old[k] for k in IDENTITY_FIELDS), tuple(new[k] for k in IDENTITY_FIELDS),
                       f'{label}.paired.rows[{index}].identity')
        results = {}
        for condition in ('normal', 'target_removed'):
            counts = {'both_correct': 0, 'both_wrong': 0, 'gained': 0, 'lost': 0}
            for old, new in zip(before, after):
                if old['condition'] != condition:
                    continue
                old_match, new_match = MATCH.search(old['output']), MATCH.search(new['output'])
                a = old_match is not None and old_match.group() == old['expected']
                b = new_match is not None and new_match.group() == new['expected']
                name = {(True, True): 'both_correct', (False, False): 'both_wrong',
                        (False, True): 'gained', (True, False): 'lost'}[(a, b)]
                counts[name] += 1
            counts.update(count=sum(counts.values()),
                          baseline_correct=counts['both_correct'] + counts['lost'],
                          adapter_correct=counts['both_correct'] + counts['gained'])
            counts['delta_percentage_points'] = 100 * (counts['gained'] - counts['lost']) / counts['count']
            for field, value in counts.items():
                stored = report['mk_comparison'][condition][field]
                if type(value) is float:
                    self.arithmetic(stored, value, label + '.mk_comparison.' + condition + '.' + field)
                else:
                    self.equal(stored, value, label + '.mk_comparison.' + condition + '.' + field)
            results[condition] = counts
        return results

    def report(self, report, label):
        self.equal(report['format'], 'MAMBA2_SOURCE_RESURFACE_EVAL_V1', label + '.format')
        self.equal(report['complete'], True, label + '.complete')
        self.equal(report['smoke'], False, label + '.smoke')
        self.equal(report['split'], 'confirm', label + '.split')
        self.equal(report['ppl_windows'], 130, label + '.ppl_windows')
        self.equal(report['mk_cases'], 768, label + '.mk_cases')
        self.equal(report['model_precision'], 'source BF16 cast to native FP16', label + '.precision')
        for key, expected in PINS.items():
            self.equal(report[key], expected, label + '.' + key)
        manifest_sha = report['data_manifest_sha256']
        fingerprint = report['dataset']['dataset_fingerprint']
        self.check(isinstance(manifest_sha, str) and SHA.fullmatch(manifest_sha) is not None,
                   label + '.data_manifest_sha256: invalid SHA256')
        self.check(isinstance(fingerprint, str) and bool(fingerprint.strip()),
                   label + '.dataset.dataset_fingerprint: expected nonempty string')
        if label == 'published':
            self.equal(manifest_sha, PUBLISHED_METADATA_PINS['data_manifest_sha256'],
                       label + '.data_manifest_sha256')
            self.equal(fingerprint, PUBLISHED_METADATA_PINS['dataset_fingerprint'],
                       label + '.dataset.dataset_fingerprint')
        for key, expected in DATASET_PINS.items():
            self.equal(report['dataset'][key], expected, label + '.dataset.' + key)
        base = report['frozen_base_check']
        self.equal(base['tensors'], 507, label + '.base.tensors')
        self.equal(base['parameters'], 8236999680, label + '.base.parameters')
        self.equal(base['identity_version_gradients_unchanged'], True, label + '.base.frozen')
        result = {}
        for arm in ('baseline', 'adapter'):
            result[arm] = {'ppl': self.ppl(report[arm + '_ppl'], label + '.' + arm + '_ppl'),
                           'mk': self.mk(report[arm + '_mk'], label + '.' + arm + '_mk')}
        for index, (a, b) in enumerate(zip(report['baseline_ppl']['windows'], report['adapter_ppl']['windows'])):
            self.equal(tuple(a[k] for k in PPL_IDENTITY_FIELDS), tuple(b[k] for k in PPL_IDENTITY_FIELDS),
                       f'{label}.paired.ppl.windows[{index}].identity')
        result['mk_comparison'] = self.paired_mk(report, label)
        delta = 100 * (result['adapter']['ppl']['ppl_recomputed'] / result['baseline']['ppl']['ppl_recomputed'] - 1)
        self.arithmetic(report['ppl_delta_percent'], delta, label + '.ppl_delta_percent')
        result['ppl_delta_percent_recomputed'] = delta
        result['restored_probe'] = self.mk(report['restored_probe'], label + '.restored_probe', probe=True)
        self.equal(report['restored_probe']['rows'], report['baseline_mk']['rows'][:8], label + '.restored_probe_exact')
        return result


def verify_reports(published, replay, actual_adapter_sha):
    audit = Audit()
    result = {
        'format': 'MAMBA2_RESURFACE_REPLAY_AUDIT_V1',
        'arithmetic_tolerance': {'relative': ARITHMETIC_REL_TOL, 'absolute': ARITHMETIC_ABS_TOL},
        'actual_adapter_sha256': actual_adapter_sha, 'pins': PINS,
        'published_metadata_pins': PUBLISHED_METADATA_PINS,
        'scope': 'Offline raw-report arithmetic, content identity, regex rescore, and fresh-report comparison. exact_replay covers scientific inputs and outputs; environment-dependent metadata equality is reported separately.',
        'limitations': [
            'The verifier does not run the model or independently recompute logits.',
            'Token hashes are compared; raw prompt/PPL token bytes are not embedded in these reports.',
            'SentencePiece decoding is not rerun. Equal generated IDs must have equal output text across reports.',
            'Original frozen-base receipt checks identities, version counters, and gradients, not full tensor contents.',
            'CONFIRM templates/instances and WikiText validation were previously observed.',
        ],
    }
    audit.equal(actual_adapter_sha, PINS['adapter_sha256'], 'actual adapter SHA256')
    for label, report in (('published', published), ('replay', replay)):
        try:
            result[label] = audit.report(report, label)
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError) as error:
            audit.errors.append(f'{label}: malformed report: {error}')
    differences = {}
    try:
        for key in PINS:
            audit.equal(published[key], replay[key], 'replay identity.' + key)
        for key in DATASET_PINS:
            audit.equal(published['dataset'][key], replay['dataset'][key], 'replay dataset identity.' + key)
        result['metadata_comparison'] = {
            'manifests_equal': published['data_manifest_sha256'] == replay['data_manifest_sha256'],
            'published_data_manifest_sha256': published['data_manifest_sha256'],
            'replay_data_manifest_sha256': replay['data_manifest_sha256'],
            'dataset_fingerprints_equal': published['dataset']['dataset_fingerprint'] == replay['dataset']['dataset_fingerprint'],
            'published_dataset_fingerprint': published['dataset']['dataset_fingerprint'],
            'replay_dataset_fingerprint': replay['dataset']['dataset_fingerprint'],
            'interpretation': 'The data manifest embeds Python version and dataset fingerprints depend on environment. These metadata may differ while all ordered prompt/token hashes, answers, dataset text/token digests, and revision remain identical.',
        }
        for arm in ('baseline', 'adapter'):
            a, b = published[arm + '_ppl'], replay[arm + '_ppl']
            for source_name, block in (('published', a), ('replay', b)):
                for field in ('nll', 'ppl'):
                    audit.number(block[field], f'{source_name}.{arm}.comparison.{field}')
            window_deltas = []
            for index, (old, new) in enumerate(zip(a['windows'], b['windows'])):
                audit.equal(tuple(old[k] for k in PPL_IDENTITY_FIELDS), tuple(new[k] for k in PPL_IDENTITY_FIELDS),
                            f'replay.{arm}.ppl.windows[{index}].identity')
                for source_name, row in (('published', old), ('replay', new)):
                    for field in ('nll', 'ppl'):
                        audit.number(row[field], f'{source_name}.{arm}.comparison.windows[{index}].{field}')
                delta = new['nll'] - old['nll']
                window_deltas.append({'index': index, 'start': old['start'], 'nll_delta': delta,
                                      'ppl_delta': new['ppl'] - old['ppl'], 'nll_exact': old['nll'] == new['nll']})
            before, after = published[arm + '_mk']['rows'], replay[arm + '_mk']['rows']
            mk_deltas = []
            for index, (old, new) in enumerate(zip(before, after)):
                audit.equal(tuple(old[k] for k in IDENTITY_FIELDS), tuple(new[k] for k in IDENTITY_FIELDS),
                            f'replay.{arm}.mk.rows[{index}].identity')
                fields = [key for key in ('generated_ids', 'output', 'prediction', 'correct') if old[key] != new[key]]
                if fields:
                    mk_deltas.append({'index': index, 'id': old['id'], 'changed_fields': fields,
                                      'published_prediction': old['prediction'], 'replay_prediction': new['prediction'],
                                      'published_correct': old['correct'], 'replay_correct': new['correct']})
            differences[arm] = {
                'ppl': {'aggregate_nll_exact': a['nll'] == b['nll'], 'aggregate_ppl_exact': a['ppl'] == b['ppl'],
                        'nll_delta': b['nll'] - a['nll'], 'ppl_delta': b['ppl'] - a['ppl'],
                        'max_abs_window_nll_error': max(abs(row['nll_delta']) for row in window_deltas),
                        'max_abs_window_ppl_error': max(abs(row['ppl_delta']) for row in window_deltas),
                        'exact_nll_windows': sum(row['nll_exact'] for row in window_deltas),
                        'windows': window_deltas},
                'mk': {'rows': len(before), 'all_generated_ids_exact': all(a['generated_ids'] == b['generated_ids'] for a, b in zip(before, after)),
                       'all_output_text_exact': all(a['output'] == b['output'] for a, b in zip(before, after)),
                       'all_predictions_exact': all(a['prediction'] == b['prediction'] for a, b in zip(before, after)),
                       'all_correctness_exact': all(a['correct'] == b['correct'] for a, b in zip(before, after)),
                       'changed_rows': len(mk_deltas), 'differences': mk_deltas},
            }
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError) as error:
        audit.errors.append('replay comparison: malformed report: ' + str(error))
    result['comparison'] = differences
    result['errors'] = audit.errors
    result['integrity_passed'] = not audit.errors
    result['exact_replay'] = not audit.errors and len(differences) == 2 and all(
        row['ppl']['aggregate_nll_exact'] and row['ppl']['aggregate_ppl_exact']
        and row['ppl']['exact_nll_windows'] == 130 and row['ppl']['max_abs_window_ppl_error'] == 0
        and row['mk']['changed_rows'] == 0 for row in differences.values())
    result['status'] = ('integrity_failure' if audit.errors else
                        'exact_replay' if result['exact_replay'] else 'valid_reports_with_replay_differences')
    return result


def self_test(published, actual_sha):
    checks = []

    def expect(name, candidate, valid, exact=False):
        result = verify_reports(published, candidate, actual_sha)
        assert result['integrity_passed'] is valid, (name, result['errors'])
        assert result['exact_replay'] is exact, name
        checks.append(name)

    expect('published versus itself', published, True, True)
    for name, mutate in (
        ('source hash tamper', lambda r: r.update(source_checkpoint_sha256='0' * 64)),
        ('malformed replay manifest hash', lambda r: r.update(data_manifest_sha256='invalid')),
        ('empty replay dataset fingerprint', lambda r: r['dataset'].update(dataset_fingerprint='')),
        ('PPL arithmetic tamper', lambda r: r['adapter_ppl'].update(nll=r['adapter_ppl']['nll'] + 1)),
        ('PPL token identity tamper', lambda r: r['adapter_ppl']['windows'][0].update(token_sha256_int64le='0' * 64)),
        ('MK regex correctness tamper', lambda r: r['adapter_mk']['rows'][0].update(correct=False)),
        ('MK duplicate identity tamper', lambda r: r['adapter_mk']['rows'][1].update(id=r['adapter_mk']['rows'][0]['id'])),
        ('MK paired counts tamper', lambda r: r['mk_comparison']['normal'].update(gained=223)),
        ('restored probe tamper', lambda r: r['restored_probe']['rows'][0].update(output='000000')),
        ('same token IDs inconsistent decode', lambda r: r['adapter_mk']['rows'][0].update(output=r['adapter_mk']['rows'][0]['output'] + ' ')),
        ('nonfinite NLL tamper', lambda r: r['adapter_ppl']['windows'][0].update(nll=float('nan'))),
    ):
        candidate = copy.deepcopy(published)
        mutate(candidate)
        expect(name, candidate, False)
    candidate = copy.deepcopy(published)
    block = candidate['adapter_ppl']
    block['windows'][0]['nll'] += 1e-7
    block['windows'][0]['ppl'] = math.exp(block['windows'][0]['nll'] / block['windows'][0]['target_tokens'])
    block['nll'] = math.fsum(row['nll'] for row in block['windows'])
    block['ppl'] = math.exp(block['nll'] / block['target_tokens'])
    candidate['ppl_delta_percent'] = 100 * (block['ppl'] / candidate['baseline_ppl']['ppl'] - 1)
    expect('consistent tiny NLL difference reported without integrity failure', candidate, True)
    candidate = copy.deepcopy(published)
    candidate['adapter_mk']['rows'][-1]['generated_ids'][-1] = 251555
    candidate['adapter_mk']['rows'][-1]['output'] += '!'
    expect('changed MK IDs and text reported without inventing decoder validation', candidate, True)
    candidate = copy.deepcopy(published)
    candidate['data_manifest_sha256'] = 'a' * 64
    candidate['dataset']['dataset_fingerprint'] = 'same-data-different-environment'
    expect('environment metadata differs but identical content and scores are accepted', candidate, True, True)
    metadata = verify_reports(published, candidate, actual_sha)['metadata_comparison']
    assert metadata['manifests_equal'] is False
    assert metadata['dataset_fingerprints_equal'] is False
    assert metadata['replay_data_manifest_sha256'] == 'a' * 64
    assert metadata['replay_dataset_fingerprint'] == 'same-data-different-environment'
    bad_sha = verify_reports(published, published, '0' * 64)
    assert not bad_sha['integrity_passed']
    checks.append('actual adapter hash tamper')
    with tempfile.TemporaryDirectory(prefix='mamba2-audit-selftest-') as directory:
        path = Path(directory) / 'receipt.json'
        write_new_audit(path, {'receipt': 'original'})
        original = path.read_bytes()
        try:
            write_new_audit(path, {'receipt': 'overwrite attempt'})
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing audit receipt was overwritten')
        assert path.read_bytes() == original
    checks.append('existing audit receipt preserved by exclusive creation')
    return {'passed': len(checks), 'checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published', type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true', help='Audit bundled published report and exercise tamper cases')
    args = parser.parse_args()
    if args.self_test:
        published = load_json(args.published or ROOT / 'reports/source_resurface_v1_confirm_full.json')
        digest = file_sha256(args.adapter or ROOT / 'artifacts/source_resurface_v1/adapter_fp16.pt')
        print(json.dumps(self_test(published, digest), indent=2, allow_nan=False))
        return 0
    if any(value is None for value in (args.published, args.replay, args.adapter, args.output)):
        parser.error('--published, --replay, --adapter, and --output are required')
    if args.output.exists():
        print(f'Refusing to overwrite existing audit receipt: {args.output}', file=sys.stderr)
        return 1
    try:
        result = verify_reports(load_json(args.published), load_json(args.replay), file_sha256(args.adapter))
        result['files'] = {key: {'path': str(path.resolve()), 'sha256': file_sha256(path)}
                           for key, path in (('published', args.published), ('replay', args.replay), ('adapter', args.adapter))}
    except (OSError, ValueError, TypeError) as error:
        result = {'format': 'MAMBA2_RESURFACE_REPLAY_AUDIT_V1', 'integrity_passed': False,
                  'exact_replay': False, 'status': 'integrity_failure', 'errors': [str(error)]}
    try:
        write_new_audit(args.output, result)
    except FileExistsError:
        print(f'Refusing to overwrite existing audit receipt: {args.output}', file=sys.stderr)
        return 1
    except OSError as error:
        print(f'Cannot write audit receipt {args.output}: {error}', file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ('status', 'integrity_passed', 'exact_replay', 'errors')},
                     indent=2, allow_nan=False))
    return 0 if result['integrity_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
