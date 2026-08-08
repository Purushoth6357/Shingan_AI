# Release Checklist

Before submitting the repository for evaluation, ensure all the following checks are complete:

- [ ] `requirements.txt` is up-to-date and dependencies are installed (`pip install -r requirements.txt`).
- [ ] `infer.py` works seamlessly from the command line on a folder of test images.
- [ ] `README.md` is complete, clear, and follows the KLA expected structure.
- [ ] Model weights/checkpoints are uploaded, linked, or included correctly without absolute paths.
- [ ] Results and metrics in `README.md` match the final trained model.
- [ ] `LICENSE` is included (if applicable).
- [ ] References (papers, repos) are properly cited.
- [ ] No absolute paths exist anywhere in the code.
- [ ] Debug prints (e.g., from `losses.py` or training loop) are removed or gated behind a debug flag.
- [ ] Local dataset paths have been generalized or use standard relative directories.
- [ ] Google Drive / Colab specific paths are removed.
- [ ] Random seed is fixed (e.g., `seed: 42`) in all configurations for reproducibility.
- [ ] Verification script (`python scripts/verify_submission.py`) passes all checks.
