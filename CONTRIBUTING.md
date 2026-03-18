# Contributing to SimAI

[中文版](CONTRIBUTING.zh-CN.md)

Thank you for your interest in contributing to SimAI! This guide will help you get started with contributing code, documentation, and ideas.

---

## What We're Building

**Vision**: The industry's first full-stack, high-precision simulator for AI large-scale inference and training.

**Goal**: Provide end-to-end modeling and simulation of AI training/inference processes—encompassing framework, collective communication, network layers, and more—so researchers can analyze performance, evaluate optimizations, and explore infrastructure designs without real hardware.

**Current Progress**: SimAI 1.5 released (Dec 2025), with end-to-end multi-request inference simulation, DeepSeek/Qwen3 model support, and Prefill/Decode separation.

**Academic Background**: Accepted by NSDI'25 Spring. See our [paper](https://arxiv.org/abs/2410.07346) for technical details.

---

## How to Contribute

### Ways to Contribute

1. **New features** — Add model support, parallelism strategies, scheduling policies, etc.
2. **Bug fixes** — Fix simulation inaccuracies, crashes, or incorrect results
3. **Performance optimization** — Improve simulation speed, memory usage, or scalability
4. **Documentation** — Improve tutorials, add examples, fix errors
5. **Benchmarks & validation** — Add validation against real hardware results
6. **Issue reports** — Report bugs, request features, or share feedback

---

## Project Architecture

SimAI is a modular project composed of 5 core submodules (Git submodules) and several supporting directories:

```
SimAI/
├── aicb/                        # AI Computation Benchmark — workload generation (Python)
│   ├── workload_generator/      #   Generates training/inference workloads
│   └── aicb.py                  #   Main entry point
├── astra-sim-alibabacloud/      # Simulation engine — core simulator (C++)
│   ├── astra-sim/               #   Extended from astra-sim 1.0
│   └── build.sh                 #   Build script
├── ns-3-alibabacloud/           # NS-3 network simulator backend (C++)
├── vidur-alibabacloud/          # LLM inference simulation (Python)
│   ├── vidur/                   #   Core simulation framework
│   └── setup.py                 #   Python package config
├── SimCCL/                      # Collective communication transformation
├── docs/                        # Documentation and tutorials
├── example/                     # Example workloads and configurations
├── scripts/                     # Build and utility scripts
│   └── build.sh                 #   Main build script
├── results/                     # Simulation output directory
├── bin/                         # Compiled binary output
├── Dockerfile                   # Docker container definition
└── README.md                    # Project documentation
```

---

## Development Environment Setup

### Prerequisites

- **Python** 3.8+ (3.12 recommended with Docker image)
- **CMake** 3.16+
- **GCC/G++** 9.4+
- **Git** with submodule support

### Option A: Docker (Recommended)

```bash
# Build the Docker image
docker build -t simai:latest .

# Run a container with GPU support
docker run --gpus all -it --rm \
    -v $(pwd)/results:/workspace/SimAI/results \
    simai:latest /bin/bash
```

### Option B: Build from Source

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/aliyun/SimAI.git
cd SimAI

# 2. Build C++ components (choose one mode)
# Analytical mode (fast, no network detail):
bash scripts/build.sh -c analytical

# NS-3 simulation mode (full-stack, detailed network):
bash scripts/build.sh -c ns3

# Physical mode (beta, RDMA clusters):
bash scripts/build.sh -c phy

# 3. Install Python dependencies
pip install -r aicb/requirements.txt
pip install -r vidur-alibabacloud/requirements.txt

# 4. Verify the build
ls bin/  # Should contain SimAI_analytical or SimAI_simulator
```

### Verify Installation

```bash
# Quick check: run a simple analytical simulation
cd bin
./SimAI_analytical \
    --workload_path=../example/workload_analytical.txt \
    --comm_group_type=TP_GROUP \
    --busbw_path=../example/busbw.yaml
```

---

## Working with Submodules

SimAI uses Git submodules for its core components. Understanding this is crucial for contributing.

### Submodule Overview

| Submodule | Repository | Language | Description |
|-----------|-----------|----------|-------------|
| `aicb` | [aliyun/aicb](https://github.com/aliyun/aicb) | Python | Workload generation |
| `SimCCL` | [aliyun/SimCCL](https://github.com/aliyun/SimCCL) | Python | Collective communication |
| `ns-3-alibabacloud` | [aliyun/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) | C++ | Network simulation |
| `astra-sim-alibabacloud` | In-tree | C++ | Simulation engine |
| `vidur-alibabacloud` | In-tree | Python | Inference simulation |

### Key Rules

1. **Submodules have independent Git histories.** Changes inside a submodule directory are tracked by that submodule's own repo, not the parent.
2. **The parent repo only tracks the commit hash** of each submodule. After modifying a submodule, you must commit in both the submodule and the parent repo.
3. **Always initialize submodules** after cloning:
   ```bash
   git submodule update --init --recursive
   ```

### Cross-Submodule Changes

If your contribution spans multiple submodules (e.g., adding a new model in `aicb` and simulation support in `astra-sim-alibabacloud`):

1. Make and commit changes in each submodule separately
2. Update the parent repo to point to the new submodule commits
3. Create separate PRs for each submodule repository if they have independent remotes
4. Reference the related PRs in your descriptions

---

## Development Workflow

### Step 1: Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone --recurse-submodules https://github.com/YOUR_USERNAME/SimAI.git
cd SimAI

# Add upstream remote
git remote add upstream https://github.com/aliyun/SimAI.git
```

### Step 2: Create a Feature Branch

```bash
# Sync with upstream first
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
# Make your changes...
# Run relevant tests (see Testing section below)

# For C++ changes, rebuild:
bash scripts/build.sh -c analytical  # or ns3

# For Python changes, verify imports and basic functionality
python -c "from aicb import ..."
```

### Step 4: Commit Your Changes

```bash
# Stage your changes
git add -A

# Commit with a descriptive message (see Commit Convention below)
git commit -m "feat(aicb): add Llama-4 model workload generation"
```

### Step 5: Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Then create a Pull Request on GitHub
```

---

## Code Style

### Python

- **Formatter**: [black](https://github.com/psf/black) (default settings)
- **Import sorting**: [isort](https://pycqa.github.io/isort/) (compatible with black)
- **Linter**: [flake8](https://flake8.pycqa.org/)
- **Max line length**: 120 characters

```bash
# Format your Python code
black --line-length 120 your_file.py
isort your_file.py
flake8 your_file.py --max-line-length 120
```

### C++

- Follow the existing code style in `astra-sim-alibabacloud/`
- Use 4-space indentation
- Keep function and variable names in `snake_case`
- Add comments for non-trivial logic

### Shell Scripts

- Use `#!/bin/bash` shebang
- Quote all variables: `"${VAR}"` not `$VAR`
- Use `set -e` for error handling where appropriate

### General Rules

- Write comments in **English**
- All new functions/classes should have docstrings or header comments
- Avoid hardcoded paths; use relative paths or configuration variables
- Keep changes focused — one feature/fix per PR

---

## Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code refactoring (no feature/fix) |
| `test` | Adding or updating tests |
| `perf` | Performance improvement |
| `chore` | Build process, tooling, dependencies |

### Scopes

`aicb`, `vidur`, `astra-sim`, `ns3`, `simccl`, `docs`, `docker`, `scripts`

### Examples

**Good:**
```
feat(aicb): add DeepSeek-V3 inference workload generation
fix(astra-sim): correct AllReduce latency calculation for ring algorithm
docs: update build instructions for NS-3 mode
perf(vidur): reduce memory allocation in request scheduler
```

**Bad:**
```
update code                          # Too vague
fix bug                              # No scope, no description
feat(aicb): Add DeepSeek-V3 inference workload generation support for the new model architecture  # Too long
```

---

## Pull Request Guidelines

### PR Title

Use the same format as commit messages: `<type>(<scope>): <description>`

### PR Description Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Change 1
- Change 2

## Testing
Describe how you tested these changes.

## Related Issues
Closes #xxx (if applicable)

## Checklist
- [ ] Code compiles without errors
- [ ] Existing simulations produce unchanged results (no precision regression)
- [ ] New code has appropriate comments
- [ ] Tests added for new functionality
- [ ] Documentation updated if needed
```

---

## Testing

### Training Simulation (Analytical Mode)

```bash
# Generate a training workload
cd aicb
python aicb.py -m training --model_name GPT-175B

# Run analytical simulation
cd ../bin
./SimAI_analytical \
    --workload_path=../example/workload_analytical.txt \
    --comm_group_type=TP_GROUP \
    --busbw_path=../example/busbw.yaml
```

### Training Simulation (NS-3 Mode)

```bash
# Build NS-3 backend first
bash scripts/build.sh -c ns3

# Run full-stack simulation
cd bin
./SimAI_simulator [simulation_parameters]
```

### Inference Simulation

```bash
# Run inference simulation via vidur
cd vidur-alibabacloud
python -m vidur.main [config_options]
```

### Verify No Regression

When modifying simulation logic, always compare results against a known-good baseline:

```bash
# Save baseline results before your changes
cp results/output_baseline.csv /tmp/baseline.csv

# After changes, compare
diff results/output_baseline.csv /tmp/baseline.csv
```

---

## Pre-Submission Quality Checklist

Before submitting your PR, run through this checklist:

```bash
# 1. C++ compilation check (if you changed C++ code)
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
git submodule status  # Ensure no unexpected submodule changes

# 5. Verify no unintended files
git diff --stat  # Review all changes before committing
```

**Checklist Summary:**
- [ ] C++ code compiles without errors or warnings
- [ ] Python code passes lint checks (black, flake8)
- [ ] Basic simulation runs successfully
- [ ] No unintended submodule pointer changes
- [ ] Commit messages follow the convention
- [ ] PR description is complete

---

## Review Process and Acceptance Criteria

### Acceptance Criteria

Your contribution will be accepted if it meets these standards:

| Criterion | Requirement |
|-----------|-------------|
| **Build** | Compiles without errors (C++ and Python) |
| **Precision** | Does not degrade existing simulation accuracy |
| **Tests** | Key code paths are covered by tests or validated |
| **Documentation** | New features have comments and/or doc updates |
| **Style** | Follows the code style guidelines above |
| **Scope** | Changes are focused and well-explained |

### Reasons for Rejection

- Build failures
- Simulation precision regression without justification
- Missing tests for new functionality
- Overly large PRs mixing unrelated changes
- Insufficient description of what/why

### Review Timeline

1. **Initial review**: Within 3-5 business days
2. **Feedback**: Constructive comments with actionable suggestions
3. **Iteration**: Address feedback and update PR
4. **Merge**: Approved PRs are merged to the main branch

---

## What NOT to Contribute

- Proprietary or closed-source dependencies
- Changes that break backward compatibility without discussion
- Large-scale reformatting changes (open an issue first)
- Untested code in simulation-critical paths
- Commits with sensitive information (API keys, internal URLs, etc.)

---

## Recognition

Contributors will be:
- Acknowledged in release notes
- Listed in project documentation
- Credited in commit history

Significant contributors may be invited to join the maintainer team.

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/aliyun/SimAI/issues)
- **Discussions**: Open an issue with "Question:" prefix
- **Documentation**: See [docs/Tutorial.md](docs/Tutorial.md) for detailed usage guides
- **Community Events**: Check [README.md](README.md) for upcoming events and workshops

---

## Thank You!

SimAI is built by a growing community of researchers and engineers. Your contributions help advance AI systems research for everyone.

**Let's build something amazing together!**
