Linting all files in the project:
```bash
pre-commit run -a
```
Tests:
```bash
# run all tests
pytest
# run tests from specific file
pytest tests/test_train.py
# run all tests except the ones marked as slow
pytest -k "not slow"