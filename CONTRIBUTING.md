# Contributing to tito-rl

Thank you for your interest in contributing to tito-rl!

## Development Setup

```bash
git clone https://github.com/kunalkarda/tito-rl.git
cd tito-rl
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

- We use `ruff` for linting and formatting.
- Run `ruff check .` and `ruff format .` before submitting PRs.

## Pull Requests

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## TITO Principle

The core invariant of this library is **never re-encode tokens you've decoded**.

Any contribution that would require re-tokenizing model-generated tokens is not aligned with the project's goals.

## Reporting Issues

Please use the GitHub issue tracker and include as much detail as possible (model, tokenizer, reproduction steps).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
