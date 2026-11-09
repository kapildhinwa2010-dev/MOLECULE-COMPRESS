# Changelog

## [0.1.0] - 2026-11-09

### Added
- Experiment A: ternary PTQ collapse measurement on Qwen2.5-0.5B.
- absmean ternarization, packed size accounting (1.6 bits/weight).
- Per-layer sensitivity scan (ternarize one layer at a time).
- int8 dynamic PTQ comparison point.
- Bounded synthetic test suite and self-diagnosing CI.
