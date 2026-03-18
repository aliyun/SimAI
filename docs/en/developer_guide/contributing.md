# Contributing to SimAI

Thank you for your interest in contributing to SimAI! This guide covers the complete development workflow.

> **Full version**: See [CONTRIBUTING.md](../../../CONTRIBUTING.md) in the project root for the comprehensive guide.

---

## Ways to Contribute

1. **New features** — Add model support, parallelism strategies, scheduling policies
2. **Bug fixes** — Fix simulation inaccuracies, crashes, or incorrect results
3. **Performance optimization** — Improve simulation speed, memory usage, or scalability
4. **Documentation** — Improve tutorials, add examples, fix errors
5. **Benchmarks & validation** — Add validation against real hardware results
6. **Issue reports** — Report bugs, request features, or share feedback

---

## Development Workflow

### Step 1: Fork and Clone

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/SimAI.git
cd SimAI
git remote add upstream https://github.com/aliyun/SimAI.git
```

### Step 2: Create a Feature Branch

```bash
git fetch upstream
git checkout -b feature/your-feature-name upstream/master

# Branch naming conventions:
#   feature/xxx  — New features
#   fix/xxx      — Bug fixes
#   docs/xxx     — Documentation
#   perf/xxx     — Performance improvements
#   refactor/xxx — Code refactoring
```

### Step 3: Develop and Test

```bash
# For C++ changes, rebuild:
bash scripts/build.sh -c analytical  # or ns3

# For Python changes:
python -c "from aicb import ..."
```

### Step 4: Commit

```bash
git add -A
git commit -m "feat(aicb): add Llama-4 model workload generation"
```

### Step 5: Push and Create PR

```bash
git push origin feature/your-feature-name
# Then create a Pull Request on GitHub
```

---

## Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code refactoring |
| `test` | Adding or updating tests |
| `perf` | Performance improvement |
| `chore` | Build process, tooling |

### Scopes

`aicb`, `vidur`, `astra-sim`, `ns3`, `simccl`, `docs`, `docker`, `scripts`

### Examples

```
feat(aicb): add DeepSeek-V3 inference workload generation
fix(astra-sim): correct AllReduce latency calculation for ring algorithm
docs: update build instructions for NS-3 mode
perf(vidur): reduce memory allocation in request scheduler
```

---

## Code Style

### Python

- **Formatter**: [black](https://github.com/psf/black) (default settings)
- **Import sorting**: [isort](https://pycqa.github.io/isort/)
- **Linter**: [flake8](https://flake8.pycqa.org/)
- **Max line length**: 120 characters

```bash
black --line-length 120 your_file.py
isort your_file.py
flake8 your_file.py --max-line-length 120
```

### C++

- Follow existing code style in `astra-sim-alibabacloud/`
- 4-space indentation
- `snake_case` for functions and variables
- Comments for non-trivial logic

### General Rules

- Write comments in **English**
- All new functions/classes should have docstrings or header comments
- Avoid hardcoded paths; use relative paths or configuration variables
- One feature/fix per PR

---

## Working with Submodules

SimAI uses Git submodules. Key points:

| Submodule | Repository | Language |
|-----------|------------|----------|
| `aicb` | [aliyun/aicb](https://github.com/aliyun/aicb) | Python |
| `SimCCL` | [aliyun/SimCCL](https://github.com/aliyun/SimCCL) | Python |
| `ns-3-alibabacloud` | [aliyun/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) | C++ |
| `astra-sim-alibabacloud` | In-tree | C++ |
| `vidur-alibabacloud` | In-tree | Python |

### Cross-Submodule Changes

If your contribution spans multiple submodules:

1. Make and commit changes in each submodule separately
2. Update the parent repo to point to new submodule commits
3. Create separate PRs for each submodule with independent remotes
4. Reference related PRs in descriptions

---

## Pull Request Guidelines

### PR Title

Use the same format as commit messages: `<type>(<scope>): <description>`

### PR Checklist

- [ ] Code compiles without errors
- [ ] Existing simulations produce unchanged results (no precision regression)
- [ ] New code has appropriate comments
- [ ] Tests added for new functionality
- [ ] Documentation updated if needed

---

## Pre-Submission Quality Checklist

```bash
# 1. C++ compilation check
bash scripts/build.sh -c analytical
bash scripts/build.sh -c ns3

# 2. Python lint check
black --check --line-length 120 your_changed_files.py
flake8 your_changed_files.py --max-line-length 120

# 3. Basic simulation test
cd bin && ./SimAI_analytical \
    --workload_path=../example/workload_analytical.txt \
    --comm_group_type=TP_GROUP \
    --busbw_path=../example/busbw.yaml

# 4. Submodule state check
git submodule status
```

---

## Acceptance Criteria

| Criterion | Requirement |
|-----------|-------------|
| **Build** | Compiles without errors |
| **Precision** | No existing simulation accuracy degradation |
| **Tests** | Key code paths are covered |
| **Documentation** | New features have comments/doc updates |
| **Style** | Follows code style guidelines |
| **Scope** | Changes are focused and well-explained |

---

## Review Timeline

1. **Initial review**: 3-5 business days
2. **Feedback**: Constructive comments with actionable suggestions
3. **Iteration**: Address feedback and update PR
4. **Merge**: Approved PRs merged to main branch

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/aliyun/SimAI/issues)
- **Discussions**: Open an issue with "Question:" prefix
- **Documentation**: See [Tutorial](../../../docs/Tutorial.md) for usage guides
