# Contributing to AtmosGen

Thanks for your interest in contributing. This guide covers the basics.

---

## Development Setup

1. **Fork and clone** the repository.

2. **Backend** -- create a virtual environment and install dependencies:

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Frontend** -- install Node.js dependencies:

   ```bash
   cd frontend
   npm install
   ```

4. Copy `.env.example` to `.env` in both `backend/` and `frontend/` and adjust values as needed.

---

## Branching

- Create a feature branch from `main`: `git checkout -b feature/your-feature`
- Keep commits focused. One logical change per commit.

---

## Code Style

- **Python**: Follow PEP 8. Use type hints where practical.
- **TypeScript/React**: Follow the existing project conventions. Use functional components and hooks.
- Keep functions short. Write docstrings for public functions.

---

## Submitting a Pull Request

1. Make sure the backend starts without errors (`python main.py`).
2. Make sure the frontend builds cleanly (`npm run build`).
3. Run any relevant tests in `backend/tests/`.
4. Open a PR against `main` with a clear description of what changed and why.

---

## Reporting Issues

Open a GitHub issue with:
- A clear title describing the problem.
- Steps to reproduce.
- Expected vs. actual behavior.
- Your environment (OS, Python version, Node version).
