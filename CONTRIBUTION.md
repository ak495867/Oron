# Contributing to Oron

So you want to contribute to a biologically-inspired memory state machine. Respectable choice.

Oron is a cognitive architecture for stateless LLMs — contributions that improve mathematical rigor, security robustness, retrieval quality, or integration breadth are the highest priority. Low-effort PRs will be composted into the episodic store and forgotten via exponential decay.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Priority Contribution Areas](#priority-contribution-areas)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- A genuine interest in how memory works (optional but appreciated)

### Local Setup

```bash
git clone https://github.com/ak495867/Oron
cd Oron
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

### Running Tests

```bash
pytest tests/
```

All tests must pass before submitting a PR. Oron has a good memory — it will remember if you broke something.

---

---

## Priority Contribution Areas

These are the areas where your time will have the most impact:

### 1. New LLM Adapters
The most wanted contributions. If you use a provider not listed here, build the adapter:
- `AnthropicAdapter` (Claude)
- `GeminiAdapter` (Google Generative AI)
- `OllamaAdapter` (local LLaMA, Mistral, Phi, etc.)
- `OpenAIAdapter` (GPT-4o and beyond)

All adapters must implement `BaseAdapter` and support both sync and async (`chat` / `achat`). If it doesn't work async, it doesn't ship.

### 2. MMR Fusion Optimization
Maximal Marginal Relevance ranking runs across all three stores before every context assembly. If you can make it faster, smarter, or more mathematically principled — open a PR.

### 3. Advanced Intent Classification
The Memory Sandbox separates content from adversarial intent. Current classification is deterministic and solid, but subtle injection attacks are creative. If you can break it, you can probably also fix it.

### 4. Consolidation Worker Improvements
The background worker clusters episodic memories and promotes patterns into the Semantic Store — mirroring sleep consolidation cycles. Better clustering, smarter promotion thresholds, lower overhead. All fair game.

### 5. Decay Model Variants
Current decay: `Salience = Importance × e^(−λ × Δt)`. Power-law variants, context-aware lambda tuning, and user-configurable decay profiles are all interesting directions. Bonus points if you bring citations.

### 6. Examples and Documentation
Runnable, real-world examples for specific use cases — customer support agents, research assistants, long-session companions — are genuinely valuable. Clear docs outlying ambiguous aspects of the architecture are equally welcome.

---

## Development Workflow

1. **Fork** the repository
2. **Create a feature branch** from `main`:
```bash
   git checkout -b feature/your-feature-name
```
3. **Make your changes** with tests
4. **Run the full test suite:**
```bash
   pytest tests/
```
5. **Submit a Pull Request** against `main`

Do not commit directly to `main`. All changes go through PR review. Yes, this includes you.

---

## Code Standards

- **Style:** PEP 8. Use `black` for formatting, `isort` for imports. Non-negotiable.
- **Type hints:** All public functions and methods must be annotated.
- **Docstrings:** Google-style. All public classes and methods.
- **Async:** Every I/O-bound operation needs an async path. Sync-only contributions for network or disk operations will be sent back.
- **Exceptions:** Raise explicit, descriptive exceptions. Bare `except` blocks are a war crime.

### Formatting

```bash
black oron/
isort oron/
```

Run this before every commit. Your future reviewers will thank you.

---

## Testing Requirements

- New features → unit tests in `tests/`
- New adapters → initialization test, sync chat test, async chat test, at minimum
- Security changes → adversarial test cases. If you touch the Memory Sandbox, prove it still holds against injection. "IGNORE ALL PREVIOUS INSTRUCTIONS" is the minimum bar, not the ceiling.
- Happy-path-only test suites will be rejected. Edge cases are where bugs live.

---

## Submitting a Pull Request

A good PR description answers three questions:

- **What** does this change?
- **Why** does it need to change?
- **How** does your approach work?

Also confirm: all existing tests pass, new tests are included, and any public API changes are clearly called out.

PRs without a description will be closed. Oron remembers, and so do we.

---

## Reporting Issues

A useful bug report includes:

- Oron version (`pip show oron`)
- Python version and OS
- Minimal reproducible example
- Full traceback

Feature requests are welcome. Frame them as a problem statement. "It would be cool if..." is a conversation starter, not an issue.

---

*Built for the next generation of stateful AI. Contributions that push that standard forward — however small — are always welcome. Even Lelouch started somewhere.*
