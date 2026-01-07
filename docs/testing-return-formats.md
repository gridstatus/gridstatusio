# Testing Return Formats

This guide shows how to test the different return formats with various dependency configurations.

## Setup

All commands assume you're in the repository root directory.

**API Key**: Export your API key before running tests:

```bash
export GRIDSTATUS_API_KEY="your_api_key_here"
# Or source from .env file:
export $(grep GRIDSTATUS_API_KEY .env | xargs)
```

## Happy Paths

### Pandas Format (default)

```bash
uv run python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient()
data = client.get_dataset('isone_fuel_mix', limit=5, return_format='pandas', verbose=False)
print(f'Type: {type(data)}')
print(data)
"
```

### Polars Format

```bash
uv run python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient()
data = client.get_dataset('isone_fuel_mix', limit=5, return_format='polars', verbose=False)
print(f'Type: {type(data)}')
print(data)
"
```

### Python Format (list of dicts)

```bash
uv run python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient()
data = client.get_dataset('isone_fuel_mix', limit=5, return_format='python', verbose=False)
print(f'Type: {type(data)}')
print(data[0])
"
```

### Using the Enum Directly

```bash
uv run python -c "
from gridstatusio import GridStatusClient, ReturnFormat
client = GridStatusClient(return_format=ReturnFormat.POLARS)
data = client.get_dataset('isone_fuel_mix', limit=5, verbose=False)
print(f'Type: {type(data)}')
"
```

## Client-Level Default vs Per-Call Override

```bash
uv run python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient(return_format='python')  # default to python
print(f'Client default: {client.return_format}')

# Override per-call to pandas
data = client.get_dataset('isone_fuel_mix', limit=3, return_format='pandas', verbose=False)
print(f'Per-call override type: {type(data)}')
"
```

## Error Cases

### Invalid Format String

```bash
uv run python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient(return_format='invalid')
"
```

Expected output:
```
ValueError: 'invalid' is not a valid ReturnFormat
```

### MissingDependencyError Message

```bash
uv run python -c "
from gridstatusio import MissingDependencyError
err = MissingDependencyError('pandas', 'pandas')
print(err)
"
```

Expected output:
```
The 'pandas' library is required for return_format='pandas'. Install it with: pip install gridstatusio[pandas]
```

## Testing with Different Dependency Configurations

These tests create local venvs to test different dependency combinations.

### Minimal Install (no pandas, no polars)

```bash
# Create venv with minimal install
uv venv .venv-minimal
uv pip install -e . --python .venv-minimal/bin/python

# Test: should default to python format
.venv-minimal/bin/python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient()
print(f'Default format: {client.return_format}')
data = client.get_dataset('isone_fuel_mix', limit=3, verbose=False)
print(f'Type: {type(data)}')
print(data[0])
"

# Test: requesting pandas should raise MissingDependencyError
.venv-minimal/bin/python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient(return_format='pandas')
"

# Test: requesting polars should raise MissingDependencyError
.venv-minimal/bin/python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient(return_format='polars')
"

# Cleanup
rm -rf .venv-minimal
```

### Pandas Only Install

```bash
# Create venv with pandas only
uv venv .venv-pandas
uv pip install -e ".[pandas]" --python .venv-pandas/bin/python

# Test: should default to pandas format
.venv-pandas/bin/python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient()
print(f'Default format: {client.return_format}')
data = client.get_dataset('isone_fuel_mix', limit=3, verbose=False)
print(f'Type: {type(data)}')
"

# Test: python format should still work
.venv-pandas/bin/python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient(return_format='python')
data = client.get_dataset('isone_fuel_mix', limit=3, verbose=False)
print(f'Type: {type(data)}')
"

# Test: requesting polars should raise MissingDependencyError
.venv-pandas/bin/python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient(return_format='polars')
"

# Cleanup
rm -rf .venv-pandas
```

### Polars Only Install

```bash
# Create venv with polars only
uv venv .venv-polars
uv pip install -e ".[polars]" --python .venv-polars/bin/python

# Test: should default to python format (pandas not available)
.venv-polars/bin/python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient()
print(f'Default format: {client.return_format}')
"

# Test: polars format should work
.venv-polars/bin/python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient(return_format='polars')
data = client.get_dataset('isone_fuel_mix', limit=3, verbose=False)
print(f'Type: {type(data)}')
"

# Test: requesting pandas should raise MissingDependencyError
.venv-polars/bin/python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient(return_format='pandas')
"

# Cleanup
rm -rf .venv-polars
```

### Full Install (pandas and polars)

```bash
# Create venv with all dependencies
uv venv .venv-all
uv pip install -e ".[all]" --python .venv-all/bin/python

# Test: should default to pandas format
.venv-all/bin/python -c "
from gridstatusio import GridStatusClient
client = GridStatusClient()
print(f'Default format: {client.return_format}')
"

# Test: all formats should work
.venv-all/bin/python -c "
from gridstatusio import GridStatusClient

client = GridStatusClient()

pandas_data = client.get_dataset('isone_fuel_mix', limit=3, return_format='pandas', verbose=False)
print(f'Pandas: {type(pandas_data)}')

polars_data = client.get_dataset('isone_fuel_mix', limit=3, return_format='polars', verbose=False)
print(f'Polars: {type(polars_data)}')

python_data = client.get_dataset('isone_fuel_mix', limit=3, return_format='python', verbose=False)
print(f'Python: {type(python_data)}')
"

# Cleanup
rm -rf .venv-all
```

## Quick Verification Script

Run all happy paths in current environment:

```bash
uv run python -c "
from gridstatusio import GridStatusClient, ReturnFormat

client = GridStatusClient()
print(f'Default format: {client.return_format}')
print()

for fmt in ['pandas', 'polars', 'python']:
    try:
        data = client.get_dataset('isone_fuel_mix', limit=2, return_format=fmt, verbose=False)
        print(f'{fmt}: {type(data).__name__} with {len(data)} rows')
    except Exception as e:
        print(f'{fmt}: ERROR - {e}')
"
```
