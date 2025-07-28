# GridStatus.io Examples

This directory contains Jupyter notebooks demonstrating how to use the GridStatus.io API to retrieve and analyze electricity grid data.

## Quick Start

### Option 1: Automated Setup (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kmax12/gridstatusio.git
   cd gridstatusio
   ```

2. **Run the setup script:**
   ```bash
   python setup.py
   ```

3. **Start Jupyter:**
   ```bash
   uv run jupyter lab
   ```

4. **Open Getting Started.ipynb** and select the "GridStatus.io" kernel

### Option 2: Manual Setup

1. **Install uv:**
   ```bash
   pip install uv
   ```

2. **Sync dependencies:**
   ```bash
   uv sync
   ```

3. **Install Jupyter kernel:**
   ```bash
   uv run python -m ipykernel install --user --name=gridstatusio --display-name="GridStatus.io"
   ```

4. **Start Jupyter:**
   ```bash
   uv run jupyter lab
   ```

## Available Examples

- **Getting Started.ipynb** - Basic introduction to the API
- **ERCOT Pricing Data.ipynb** - Working with ERCOT settlement point prices
- **CAISO April Net Load.ipynb** - Analyzing CAISO net load data
- **Stacked Net Load Visualization.ipynb** - Creating stacked visualizations
- **Resample Data.ipynb** - Resampling and aggregating data
- **ISO Hubs.ipynb** - Working with ISO hub data

## Getting Your API Key

1. Create an account at [GridStatus.io](https://www.gridstatus.io)
2. Go to [Settings > API](https://www.gridstatus.io/settings/api)
3. Generate your API key
4. Set it as an environment variable:
   ```bash
   export GRIDSTATUS_API_KEY="your_key_here"
   ```
   Or pass it directly to the client in the notebook.

## Troubleshooting

### Common Issues

1. **"uv command not found"**
   - Install uv: `pip install uv`
   - Restart your terminal

2. **Import errors in notebooks**
   - Make sure you're using the "GridStatus.io" kernel
   - Try restarting the kernel after setup

3. **Permission errors**
   - Use `--user` flag for kernel installation
   - Or run with sudo (macOS/Linux)

### Need Help?

- Check the [Setup Guide](Setup_Guide.md) for detailed instructions
- Visit the [GridStatus.io API documentation](https://www.gridstatus.io/api)
- Open an issue on [GitHub](https://github.com/kmax12/gridstatusio/issues) 