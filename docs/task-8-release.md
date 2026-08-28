# Task 8: Open Source Release

## Overview

Prepare the project for open source release with proper licensing, documentation, packaging, and community contribution guidelines.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Open Source Release Structure                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Repository Structure                          │   │
│  │                                                                   │   │
│  │  inference-gateway/                                               │   │
│  │  ├── README.md              # Main documentation                  │   │
│  │  ├── LICENSE                # MIT License                         │   │
│  │  ├── CONTRIBUTING.md        # Contribution guidelines             │   │
│  │  ├── CODE_OF_CONDUCT.md     # Community standards                 │   │
│  │  ├── pyproject.toml         # Package configuration               │   │
│  │  ├── requirements.txt       # Dependencies                        │   │
│  │  ├── router.yaml            # Default configuration               │   │
│  │  ├── .github/                                                       │   │
│  │  │   ├── workflows/                                                 │   │
│  │  │   │   └── ci.yml         # CI/CD pipeline (Task 1)              │   │
│  │  │   ├── ISSUE_TEMPLATE/                                          │   │
│  │  │   │   ├── bug_report.md                                      │   │
│  │  │   │   └── feature_request.md                                 │   │
│  │  │   └── PULL_REQUEST_TEMPLATE.md                               │   │
│  │  ├── docs/                                                        │   │
│  │  │   ├── task-1-cicd.md       # CI/CD guide                      │   │
│  │  │   ├── task-2-testing.md    # Testing guide                    │   │
│  │  │   ├── task-3-rationale.md  # Rationale capture                │   │
│  │  │   ├── task-4-benchmarks.md # Benchmark suites                 │   │
│  │  │   ├── task-5-cache.md      # Cache enhancements               │   │
│  │  │   ├── task-6-backend.md    # Backend abstraction              │   │
│  │  │   ├── task-7-refinement.md # Refinement loop                  │   │
│  │  │   └── task-8-release.md    # This file                        │   │
│  │  ├── proxy.py               # Main gateway code                   │   │
│  │  ├── backend_interface.py   # Abstract backend (Task 6)          │   │
│  │  ├── backends/                                                    │   │
│  │  │   ├── __init__.py                                            │   │
│  │  │   ├── omlx.py                                                │   │
│  │  │   ├── mlx.py                                                 │   │
│  │  │   └── llama.py                                               │   │
│  │  ├── refinement/                                                  │   │
│  │  │   ├── __init__.py                                            │   │
│  │  │   ├── analysis.py                                            │   │
│  │  │   └── generator.py                                           │   │
│  │  └── tests/                                                       │   │
│  │      ├── __init__.py                                            │   │
│  │      ├── conftest.py                                            │   │
│  │      ├── test_proxy.py                                          │   │
│  │      ├── test_routing.py                                        │   │
│  │      ├── test_cache.py                                          │   │
│  │      ├── test_memory.py                                         │   │
│  │      ├── test_benchmarks.py                                     │   │
│  │      ├── test_backends.py                                       │   │
│  │      ├── test_refinement.py                                     │   │
│  │      └── integration/                                           │   │
│  │          └── test_e2e.py                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Agentic Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agentic Development Pipeline                        │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  Developer  │───▶│  Tester     │───▶│  Engineer   │                │
│  │  Agent      │    │  Agent      │    │  Agent      │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        ▼                    ▼                    ▼                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  Create     │    │  Verify     │    │  Publish    │                │
│  │  docs +     │    │  all tasks  │    │  to PyPI +  │                │
│  │  license +  │    │  complete   │    │  GitHub     │                │
│  │  packaging  │    │             │    │             │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        └────────────────────┴────────────────────┘                      │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  QA Agent       │                                │
│                     │  (final review, │                                │
│                     │   smoke test)   │                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Developer** | Create docs, license, packaging config | `write`, `edit` |
| **Tester** | Verify all tasks complete, run full test suite | `bash` (pytest, curl) |
| **Engineer** | Publish to PyPI, create GitHub release | `bash` (git, twine) |
| **QA** | Final review, smoke test, verify docs | `read`, `bash` |

## Step-by-Step Execution

### Step 1: Add License

**LICENSE** (new file):

```
MIT License

Copyright (c) 2026 Akshay Shah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Step 2: Create Contribution Guidelines

**CONTRIBUTING.md** (new file):

```markdown
# Contributing to Inference Gateway

Thank you for considering contributing to the Inference Gateway!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/inference-gateway.git
   cd inference-gateway
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   pytest tests/ -v
   ```
4. Run linting:
   ```bash
   ruff check .
   ```
5. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add your feature"
   ```
6. Push and create a pull request

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions
- Keep lines under 100 characters

## Testing

- Write unit tests for all new features
- Aim for >80% code coverage
- Test edge cases and error conditions
- Use mocks for external dependencies

## Pull Request Process

1. Update README.md if needed
2. Ensure all tests pass
3. Request a review
4. Address feedback
5. Merge when approved
```

### Step 3: Create Code of Conduct

**CODE_OF_CONDUCT.md** (new file):

```markdown
# Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone.

## Our Standards

Examples of behavior that contributes to creating a positive environment:
- Being respectful of differing opinions, viewpoints, and experiences
- Giving and gracefully accepting constructive feedback
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior:
- Trolling, insulting or derogatory comments
- Public or private harassment
- Publishing others' private information without consent
- Other conduct which could reasonably be considered inappropriate

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the community leaders. All complaints will be reviewed and
investigated promptly.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org/).
```

### Step 4: Update pyproject.toml for Packaging

**pyproject.toml** (update):

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "inference-gateway"
version = "2.1.1"
description = "Local inference gateway for Apple Silicon with intelligent model routing"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.12"
authors = [
    {name = "Akshay Shah", email = "akshay@example.com"}
]
keywords = ["llm", "inference", "gateway", "mlx", "apple-silicon"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "httpx>=0.28",
    "psutil>=6.0",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.3",
    "mypy>=1.8",
    "pre-commit>=3.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/inference-gateway"
Documentation = "https://github.com/yourusername/inference-gateway#readme"
Repository = "https://github.com/yourusername/inference-gateway"
Issues = "https://github.com/yourusername/inference-gateway/issues"

[tool.setuptools]
packages = ["backends", "refinement"]
```

### Step 5: Create GitHub Issue Templates

**.github/ISSUE_TEMPLATE/bug_report.md**:

```markdown
---
name: Bug Report
about: Report a bug or issue
title: ''
labels: bug
assignees: ''
---

## Description
Describe the bug you encountered.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
Describe what you expected to happen.

## Actual Behavior
Describe what actually happened.

## Environment
- OS: [e.g., macOS 15.0]
- Python version: [e.g., 3.12]
- Gateway version: [e.g., 2.1.1]
- Backend: [e.g., oMLX, MLX, LLaMA]

## Logs
Paste any relevant logs here.

## Additional Context
Add any other context about the problem here.
```

**.github/ISSUE_TEMPLATE/feature_request.md**:

```markdown
---
name: Feature Request
about: Suggest a new feature
title: ''
labels: enhancement
assignees: ''
---

## Description
Describe the feature you would like to see.

## Motivation
Why is this feature important? What problem does it solve?

## Proposed Solution
Describe the solution you would like.

## Alternatives Considered
Describe any alternative solutions or features you've considered.

## Additional Context
Add any other context or screenshots here.
```

**.github/PULL_REQUEST_TEMPLATE.md**:

```markdown
## Summary
Brief description of the changes.

## Changes
- [ ] Change 1
- [ ] Change 2

## Testing
Describe how you tested your changes.

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or breaking changes documented)
```

### Step 6: Update README for Open Source

**README.md** — Add open source sections:

```markdown
## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [x] Core gateway with judge routing
- [x] Memory protection
- [x] Response caching
- [x] Benchmark lab
- [ ] Semantic cache deduplication
- [ ] Generic backend abstraction
- [ ] Instruction refinement loop
- [ ] Plugin system for custom backends

## Community

- [Issues](https://github.com/yourusername/inference-gateway/issues) - Report bugs and request features
- [Discussions](https://github.com/yourusername/inference-gateway/discussions) - Ask questions and share ideas
```

### Step 7: Create Release Checklist

**RELEASE_CHECKLIST.md** (new file):

```markdown
# Release Checklist

## Pre-Release

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Linting passes: `ruff check .`
- [ ] Type checking passes: `mypy proxy.py --ignore-missing-imports`
- [ ] Documentation is up to date
- [ ] CHANGELOG.md updated
- [ ] Version number incremented in pyproject.toml
- [ ] All open PRs reviewed and merged
- [ ] No known critical bugs

## Release

- [ ] Create release branch: `git checkout -b release/vX.Y.Z`
- [ ] Tag the release: `git tag vX.Y.Z`
- [ ] Push tag: `git push origin vX.Y.Z`
- [ ] Create GitHub Release with notes
- [ ] Build and upload to PyPI:
  ```bash
  pip install build twine
  python -m build
  twine upload dist/*
  ```

## Post-Release

- [ ] Verify PyPI package installs: `pip install inference-gateway`
- [ ] Smoke test installed package
- [ ] Announce release (Twitter, blog, etc.)
- [ ] Monitor for issues in first 24 hours
```

### Step 8: Verify Locally

```bash
# Run full test suite
pytest tests/ -v --cov=proxy --cov-report=term-missing

# Check linting
ruff check .

# Check types
mypy proxy.py --ignore-missing-imports

# Build package
python -m build

# Test install
pip install dist/inference_gateway-2.1.1-py3-none-any.whl

# Verify installation
python -c "import proxy; print('Gateway imported successfully')"
```

### Step 9: Publish

```bash
# Create GitHub release
git tag v2.1.1
git push origin v2.1.1

# Create release on GitHub
# Go to: https://github.com/yourusername/inference-gateway/releases/new
# Select tag: v2.1.1
# Write release notes
# Publish release

# Upload to PyPI
pip install build twine
python -m build
twine upload dist/*
```

## Success Criteria

- [ ] MIT License added
- [ ] CONTRIBUTING.md created
- [ ] CODE_OF_CONDUCT.md created
- [ ] pyproject.toml configured for packaging
- [ ] GitHub issue templates created
- [ ] PR template created
- [ ] README updated for open source
- [ ] Release checklist created
- [ ] Package builds successfully
- [ ] Package installs and runs
- [ ] GitHub release created
- [ ] PyPI package published (optional)

## Commands Reference

```bash
# Run full test suite
pytest tests/ -v --cov=proxy --cov-report=term-missing

# Check linting
ruff check .

# Check types
mypy proxy.py --ignore-missing-imports

# Build package
python -m build

# Test install
pip install dist/inference_gateway-2.1.1-py3-none-any.whl

# Create GitHub release
git tag v2.1.1
git push origin v2.1.1

# Upload to PyPI
pip install build twine
python -m build
twine upload dist/*
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check pyproject.toml syntax |
| PyPI upload fails | Check TWINE_USERNAME/TWINE_PASSWORD |
| Package doesn't install | Verify dependencies in pyproject.toml |
| Tests fail after install | Check for missing files in package |
