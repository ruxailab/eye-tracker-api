# Contributing to Eye Lab: Gaze Tracker API

Thank you for your interest in contributing! 🎉  
Eye Lab is an open-source eye-tracking backend built with Python and Flask. Every contribution — big or small — is welcome.

---

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Setting Up Locally](#setting-up-locally)
- [How to Contribute](#how-to-contribute)
- [Branch Naming](#branch-naming)
- [Commit Message Style](#commit-message-style)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Issue Guidelines](#issue-guidelines)
- [Code Style](#code-style)

---

## Getting Started

1. **Fork** this repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/eye-tracker-api.git
   cd eye-tracker-api
   ```
3. **Add upstream remote** to stay in sync:
   ```bash
   git remote add upstream https://github.com/ruxailab/eye-tracker-api.git
   ```

---

## Setting Up Locally

### Prerequisites
- Python 3.x
- pip

### Steps

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env and fill in your values

# Run the Flask app
flask run
```

> **Note:** The project has a frontend counterpart at [web-eye-tracker-front](https://github.com/ruxailab/web-eye-tracker-front). Install it as well for the full application experience.

---

## How to Contribute

### Working on an existing issue
1. Go to the [Issues](https://github.com/ruxailab/eye-tracker-api/issues) tab
2. Find an unassigned issue you want to work on
3. **Comment on it** expressing your interest — the first person to comment gets preference
4. Wait for a maintainer to assign it to you before starting work

### Raising a new issue
1. Check that the issue doesn't already exist
2. Open a new issue with a clear title and description
3. The person who raises an issue has **preference** to fix it
4. Tag it appropriately (`bug`, `enhancement`, `documentation`, etc.)

---

## Branch Naming

Use the following convention:

| Type | Format | Example |
|------|--------|---------|
| Bug fix | `fix/<short-description>` | `fix/hardcoded-file-paths` |
| New feature | `feat/<short-description>` | `feat/auto-model-selection` |
| Documentation | `docs/<short-description>` | `docs/add-contributing-guide` |
| Refactor | `refactor/<short-description>` | `refactor/rename-typo-function` |
| Tests | `test/<short-description>` | `test/add-predict-unit-tests` |

---

## Commit Message Style

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>

# Examples:
fix: remove hardcoded Firebase URL and use environment variable
feat: add auto model selection for gaze estimation
docs: add CONTRIBUTING.md and .env.example
refactor: rename trian_and_predict to train_and_predict
test: add smoke test for Flask app startup
```

**Types:** `fix`, `feat`, `docs`, `refactor`, `test`, `chore`, `perf`

---

## Pull Request Guidelines

1. **One PR per issue** — keep your changes focused
2. **Link the issue** in your PR description using `Closes #<issue-number>`
3. **Describe your changes** clearly — what you changed and why
4. **Test your changes** locally before submitting
5. **Keep PRs small** — large PRs are harder to review and less likely to be merged quickly

### PR Description Template

```markdown
## What does this PR do?
<Brief description of the change>

## Related Issue
Closes #<issue-number>

## Changes Made
- <change 1>
- <change 2>

## How to Test
<Steps to verify the change works>
```

---

## Issue Guidelines

When opening an issue, please include:

- **Clear title** summarizing the problem or proposal
- **Description** of what's happening vs. what's expected
- **Steps to reproduce** (for bugs)
- **Proposed solution** (if you have one)
- **Your environment** (OS, Python version) for bugs

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code
- Use meaningful variable and function names
- Add docstrings to new functions
- Remove unused imports before submitting

---

## Need Help?

- Join our [Discord](https://discord.gg/fDFwcE6WD2) and introduce yourself
- Ask questions in [GitHub Discussions](https://github.com/ruxailab/RUXAILAB/discussions)
- Email us at `ruxailab@gmail.com`

We look forward to your contributions! 🚀