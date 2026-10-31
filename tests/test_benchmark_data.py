"""Official metric edge cases and exact middle-truncation semantics; no downloads."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from integrations.benchmark_data import (
    qa_f1_score, read_samples, retrieval_score, score, tokenize_prompt, tokenizer_provenance,
)
from integrations.ruler_data import command, score as ruler_score
from integrations.sala_validation import (
    logprobs_equal, public_provenance, selector_evidence, validate_fraction,
)


class CharacterTokenizer:
    def __call__(self, text, truncation=False):
        return {"input_ids": [0] + [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i)


class BenchmarkDataTests(unittest.TestCase):
    def test_public_tokenizer_provenance_does_not_publish_model_directory(self):
        class Tokenizer:
            name_or_path = "C:/private-user/checkpoints/model"

            def get_vocab(self):
                return {"hello": 0, "world": 1}

        provenance = tokenizer_provenance(Tokenizer())
        self.assertEqual(provenance['tokenizer'], 'local-tokenizer')
        self.assertEqual(len(provenance['tokenizer_vocab_sha256']), 64)
        self.assertNotIn('private-user', json.dumps(provenance))
        legacy = {'tokenizer': 'C:/private-user/model',
                  'nested': ['/home/private-user/model'],
                  'source_url': 'https://example.org/official'}
        cleaned = public_provenance(legacy)
        self.assertNotIn('private-user', json.dumps(cleaned))
        self.assertEqual(cleaned['source_url'], legacy['source_url'])

    def test_sala_missing_hook_or_zero_selector_is_not_a_verified_run(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'adapter.jsonl'
            for content in (None, '', json.dumps({'run_id': 'old-request'})):
                if content is not None:
                    log.write_text(content, encoding='utf-8')
                with self.assertRaises(RuntimeError):
                    selector_evidence(log, 0, 'current', 'content', 'capture', 16384, .25)

    def test_sala_evidence_binds_request_layer_work_and_effective_cut(self):
        event = dict(run_id='run', identity='content', mode='replay', layer=3,
                     block_size=64, effective_head=4096, query_tokens=16384,
                     native_selector_queries=4096, native_selector_called=True)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'adapter.jsonl'
            def validate(value, **kwargs):
                log.write_text(json.dumps(value) + '\n', encoding='utf-8')
                return selector_evidence(log, 0, 'run', 'content', 'replay', 16384, .25,
                                         **kwargs)
            self.assertTrue(validate(event, expected_layers=[3])['adapter_interception_verified'])
            for replacement in ({'identity': 'other'}, {'effective_head': 8192},
                                {'native_selector_queries': 0}, {'query_tokens': 8192},
                                {'block_size': None}, {'native_selector_called': False}):
                with self.assertRaises(RuntimeError):
                    validate(dict(event, **replacement))
            with self.assertRaises(RuntimeError):
                validate(event, expected_layers=[3, 4])
            # A full cached replay has real intercepted layer events despite zero selector work.
            zero = dict(event, effective_head=0, native_selector_queries=0,
                        native_selector_called=False)
            log.write_text(json.dumps(zero) + '\n', encoding='utf-8')
            self.assertTrue(selector_evidence(log, 0, 'run', 'content', 'replay',
                                             16384, 0)['adapter_interception_verified'])

    def test_sala_absent_logprobs_and_invalid_fractions_are_not_success(self):
        with self.assertRaises(RuntimeError):
            logprobs_equal({}, {})
        observed = {'meta_info': {'output_token_logprobs': [[-1, 7, None]]}}
        self.assertTrue(logprobs_equal(observed, observed))
        for fraction in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                validate_fraction(fraction)

    def test_ruler_official_task_configuration_and_metric(self):
        self.assertEqual(ruler_score("niah_single_1", "1234567 extra", ["1234567"]), 1)
        self.assertEqual(ruler_score("niah_multikey_2", "AB", ["ab", "cd"]), 0.5)
        args = command(Path("official"), Path("tokenizer"), Path("cache"),
                       "niah_multikey_2", 16384, 4, "official template")
        self.assertEqual(args[args.index("--type_haystack") + 1], "needle")
        self.assertEqual(args[args.index("--tokens_to_generate") + 1], "128")
        self.assertEqual(args[-1], "official template")

    def test_official_metrics(self):
        self.assertEqual(qa_f1_score("The CAT!", "a cat"), 1)
        self.assertEqual(qa_f1_score("", ""), 0)
        self.assertAlmostEqual(qa_f1_score("cat cat", "cat"), 2 / 3)
        self.assertEqual(retrieval_score("Paragraph 3, or 4", "Paragraph 3"), 0.5)
        self.assertEqual(retrieval_score("03", "Paragraph 3"), 0)
        self.assertEqual(retrieval_score("unknown", "Paragraph 3"), 0)
        self.assertEqual(score("2wikimqa", "second", ["first", "second"]), 1)

    def test_truncation_includes_special_tokens_and_retokenizes(self):
        text, ids, raw, truncated = tokenize_prompt(CharacterTokenizer(), "abcdefghi", 7)
        self.assertEqual((text, raw, truncated), ("abghi", 10, True))
        self.assertEqual(ids, [0, 97, 98, 103, 104, 105])
        self.assertEqual(tokenize_prompt(CharacterTokenizer(), "abc", 4),
                         ("abc", [0, 97, 98, 99], 4, False))

    def test_only_allowlisted_zip_member_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "data.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("../escaped.json", "untrusted")
                zipped.writestr("data/2wikimqa.jsonl", json.dumps({"_id": "a"}) + "\n"
                                + json.dumps({"_id": "b"}) + "\n")
            self.assertEqual(list(read_samples(archive, "2wikimqa", 1)), [(0, {"_id": "a"})])
            self.assertFalse((Path(directory).parent / "escaped.json").exists())
            with self.assertRaises(ValueError):
                list(read_samples(archive, "../escaped", 1))
            with self.assertRaises(ValueError):
                list(read_samples(archive, "2wikimqa", 0))


if __name__ == "__main__":
    unittest.main()
