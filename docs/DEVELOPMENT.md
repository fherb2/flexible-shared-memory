# Development Workflow

This guide describes how to work on the Flexible Shared Memory project, from making changes to releasing new versions.

## Table of Contents

- [Branch Strategy](#branch-strategy)
- [Version Management](#version-management)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Release Process](#release-process)

---

## Branch Strategy

### Main Branches

- **`main`**: Production-ready code, tagged releases only
- **`develop`**: Integration branch for features (optional, create when needed)

### Feature/Fix Branches

Always create a new branch for changes:

```bash
# For new features:
git checkout -b feature/add-async-support

# For bug fixes:
git checkout -b fix/string-encoding-bug

# For documentation:
git checkout -b docs/update-readme
```

**Naming convention:**
- `feature/*` - New functionality
- `fix/*` - Bug fixes
- `docs/*` - Documentation only
- `test/*` - Test improvements
- `refactor/*` - Code refactoring

---

## Version Management

### Semantic Versioning (SemVer)

We follow [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH

Example: 0.1.0 → 0.1.1 → 0.2.0 → 1.0.0
```

**When to increment:**

| Change Type | Version | Example | When to use |
|------------|---------|---------|-------------|
| **PATCH** | 0.1.0 → 0.1.1 | Bug fixes | Backward-compatible fixes |
| **MINOR** | 0.1.1 → 0.2.0 | New features | Backward-compatible additions |
| **MAJOR** | 0.2.0 → 1.0.0 | Breaking changes | API changes, incompatible updates |

**Special versions:**
- `0.x.y` - Pre-release, API not stable
- `1.0.0` - First stable release
- `1.x.y` - Stable API, production-ready

### Files to Update

**Always update both files:**

1. **`pyproject.toml`** (Line 3):
   ```toml
   version = "0.1.1"
   ```

2. **`source/flexible_shared_memory/__init__.py`** (Line 45):
   ```python
   __version__ = "0.1.1"
   ```

### Version Update Script

For convenience, use this helper:

```bash
# Update version in both files
NEW_VERSION="0.1.1"

sed -i "3s/version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
sed -i "45s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" source/flexible_shared_memory/__init__.py

# Verify changes
grep "version" pyproject.toml
grep "__version__" source/flexible_shared_memory/__init__.py
```

---

## Development Workflow

### 1. Start New Work

```bash
# Make sure main is up-to-date
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

```bash
# Edit code
nano source/flexible_shared_memory/shared_memory.py

# Run tests frequently
poetry run pytest tests/ -v

# Check coverage
poetry run pytest tests/ --cov=flexible_shared_memory --cov-report=term
```

### 3. Write Tests

**Always add tests for new features:**

```bash
# Create or update test file
nano tests/test_your_feature.py

# Run your new tests
poetry run pytest tests/test_your_feature.py -v
```

### 4. Update Version Numbers

**Decide on version bump:**

```bash
# Current version
grep "version" pyproject.toml

# Update to new version (both files!)
# See "Version Management" section above
```

### 5. Update Documentation

**If user-facing changes:**

- Update `README.md` with new examples
- Update docstrings in code
- Update `research/test-planning.md` if test strategy changes

### 6. Commit Changes

```bash
# Stage changes
git add .

# Check what will be committed
git status
git diff --cached

# Commit with descriptive message
git commit -m "feat: Add async support for read operations

- Implement async read() with asyncio
- Add async examples to README
- Add 15 new tests for async functionality
- Update version to 0.2.0"
```

**Commit message format:**

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `chore:` - Maintenance (dependencies, etc.)

### 7. Push Branch

```bash
# Push feature branch to GitHub
git push -u origin feature/your-feature-name
```

### 8. Create Pull Request (Optional)

If working in a team or want review:

1. Go to GitHub repository
2. Click "Compare & pull request"
3. Review changes
4. Create Pull Request to `main`
5. Wait for CI to pass (green checkmark)
6. Merge when ready

### 9. Merge to Main

**If working solo (direct merge):**

```bash
# Switch to main
git checkout main

# Merge feature branch
git merge feature/your-feature-name

# Push to GitHub
git push origin main
```

**After merge:**
```bash
# Delete feature branch (optional)
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

---

## Testing

### Run All Tests

```bash
# Basic test run
poetry run pytest tests/ -v

# With coverage
poetry run pytest tests/ --cov=flexible_shared_memory --cov-report=term --cov-report=html

# Single test file
poetry run pytest tests/test_write_basic.py -v

# Single test
poetry run pytest tests/test_write_basic.py::TestWriteScalars::test_write_single_float -v
```

### Before Committing

**Checklist:**
- [ ] All tests pass: `poetry run pytest tests/ -v`
- [ ] Coverage ≥ 80%: Check HTML report
- [ ] Code formatted: `poetry run black source/`
- [ ] No lint errors: `poetry run flake8 source/`
- [ ] Documentation updated
- [ ] Version numbers updated (both files)

### CI Pipeline

GitHub Actions runs automatically on push:
- Tests on Python 3.9, 3.10, 3.11, 3.12, 3.13
- Tests with NumPy 1.26 and 2.0
- 9 total test combinations

**Check status:**
- Go to: https://github.com/fherb2/flexible-shared-memory/actions
- Ensure all jobs pass (green checkmarks)

---

## Release Process

### Full Release Workflow

#### 1. Prepare Release Branch (Optional for Major Releases)

```bash
git checkout -b release/v0.2.0
```

#### 2. Final Checks

```bash
# Run all tests
poetry run pytest tests/ -v

# Check coverage
poetry run pytest tests/ --cov=flexible_shared_memory --cov-report=term

# Verify version numbers
grep "version" pyproject.toml
grep "__version__" source/flexible_shared_memory/__init__.py

# Check README is up-to-date
cat README.md
```

#### 3. Update CHANGELOG (Recommended)

Create `CHANGELOG.md` if not exists:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2024-11-27

### Added
- Async support for read operations
- New examples for async usage

### Fixed
- String encoding bug with special characters

### Changed
- Improved performance of array writes (15% faster)

## [0.1.1] - 2024-11-26

### Fixed
- Fixed timeout handling in FIFO mode

## [0.1.0] - 2024-11-26

### Added
- Initial release
- Lock-free shared memory
- Field-level status tracking
- NumPy array support
```

#### 4. Commit Release Changes

```bash
git add .
git commit -m "chore: Prepare release v0.2.0

- Update version to 0.2.0
- Update CHANGELOG
- Final documentation updates"

git push origin release/v0.2.0
```

#### 5. Merge to Main

```bash
git checkout main
git merge release/v0.2.0
git push origin main
```

#### 6. Create Git Tag

```bash
# Create annotated tag
git tag -a v0.2.0 -m "Release v0.2.0

New Features:
- Async read support
- Performance improvements

Bug Fixes:
- String encoding with special characters

Full changelog: https://github.com/fherb2/flexible-shared-memory/blob/main/CHANGELOG.md"

# Push tag to GitHub
git push origin v0.2.0
```

#### 7. Create GitHub Release

1. Go to: https://github.com/fherb2/flexible-shared-memory/releases
2. Click "Draft a new release"
3. Select tag: `v0.2.0`
4. Release title: `v0.2.0 - Async Support`
5. Description: Copy from CHANGELOG
6. Click "Publish release"

#### 8. Verify Release

- [ ] GitHub Release created
- [ ] CI passes on tag
- [ ] README badge shows passing
- [ ] Version numbers correct on GitHub

#### 9. Clean Up

```bash
# Delete release branch
git branch -d release/v0.2.0
git push origin --delete release/v0.2.0
```

---

## Quick Reference

### Common Commands

```bash
# Start new feature
git checkout -b feature/my-feature

# Run tests
poetry run pytest tests/ -v

# Update version (manual)
nano pyproject.toml  # Line 3
nano source/flexible_shared_memory/__init__.py  # Line 45

# Commit
git add .
git commit -m "feat: Description"
git push

# Merge to main
git checkout main
git merge feature/my-feature
git push

# Create release
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

### Version Bump Decision Tree

```
Breaking change? (API incompatible)
├─ Yes → MAJOR (0.x.y → 1.0.0)

- ❌ Committing directly to `main` (except for solo hotfixes)
- ❌ Pushing broken tests
- ❌ Forgetting to update version numbers
- ❌ Releasing without testing
- ❌ Unclear commit messages

### When in Doubt

```bash
# Check current status
git status
git log --oneline -5

# See what changed
git diff

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo all local changes
git reset --hard HEAD
```

---

## Contact

For questions about development workflow:
- Open an issue: https://github.com/fherb2/flexible-shared-memory/issues
- Check existing PRs: https://github.com/fherb2/flexible-shared-memory/pulls