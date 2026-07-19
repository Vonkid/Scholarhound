# ScholarHound

ScholarHound is an end-user-driven, open-source literature-intelligence system for researchers who need a dependable way to keep up with fast-moving scientific fields.

It discovers papers, scores their relevance, produces daily research digests, and records LLM-assisted judgments as explicit, provenance-tagged, revisable objects. Its central claim is deliberately narrow: ScholarHound does not make an LLM into a scientific oracle. It adds an accountability layer so model judgments can be inspected, challenged, revised, and routed to a human when they are unreliable.

> **Status:** Alpha research software. Interfaces and schemas may change. Do not use ScholarHound as a substitute for reading primary sources or for expert scientific judgment.

## Why it exists

ScholarHound grew out of daily research practice rather than a benchmark-only demo. Literature monitoring creates recurring maintenance work: feeds fail, metadata conflicts, papers duplicate across sources, relevance changes with the project, and fluent model summaries can conceal uncertainty. The project turns those failures into code, tests, provenance rules, and human-review gates.

## What it provides

- RSS, Crossref, and journal-TOC ingestion
- relevance, novelty, and bridge scoring
- configurable LLM-assisted ranking and relation extraction
- Markdown daily digests and Obsidian-compatible output
- a local HTML interface
- append-only state-change records
- a belief-centered V3 kernel with validation and human-review boundaries
- benchmark and ablation utilities

## Epistemic boundary

ScholarHound reasons over what the literature reports; it does not observe ground truth. Confidence represents the strength and consistency of the available literature signal, not the probability that a claim is true. Deterministic rules handle fields that can be parsed reliably. Contested interpretive judgments remain visible and are routed to a human instead of being silently committed.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp config.example.yaml config.yaml
export SCHOLARHOUND_API_KEY="your-key-here"
scholarhound scan --config config.yaml
```

Use a different provider, model, or environment-variable name by editing `config.yaml`. Never commit real credentials or a configuration containing private local paths.

To see the available workflows:

```bash
scholarhound --help
```

To launch the local interface:

```bash
scholarhound serve --config config.yaml
```

## Tests

```bash
python -m pytest -q
```

The public-release candidate was checked against 182 local tests before its initial publication.

## Privacy model

The repository intentionally excludes private paper libraries, real source registries, local filesystem paths, research diaries, participant feedback, unpublished manuscripts, and production infrastructure configuration. Examples intended for publication must use public or synthetic data.

## Project layout

```text
psil/                 Python package and CLI
tests/                Unit and integration tests
config.example.yaml   Safe configuration template
.github/workflows/    CI and release-safety checks
```

The internal package name remains `psil` (Personal Scientific Intelligence Layer) for compatibility. The public product and command name are `ScholarHound` and `scholarhound`.

## Publication and priority

Git commits, tags, and releases provide a public, verifiable implementation timeline. They are useful provenance, but they are not a substitute for a paper, DOI, patent filing, or formal scholarly priority mechanism. If academic priority matters, archive a tagged release with a DOI-granting repository when ready.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please avoid submitting copyrighted papers, private research corpora, personal data, credentials, or production infrastructure details.

## License

MIT. See [LICENSE](LICENSE).
