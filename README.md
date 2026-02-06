# Model Selection

A Python project for model selection and evaluation.

## Type Checking

This project uses **basedpyright** for static type checking.

### Prerequisites

basedpyright must be installed via homebrew:

```bash
brew install basedpyright
```

### Run Type Check

```bash
basedpyright src
```

**Expected output for passing**: `0 errors, X warnings, 0 notes`

The type checker will report any type errors found in the `src` directory. A successful run shows 0 errors (warnings from third-party libraries like the OpenAI SDK are acceptable).
