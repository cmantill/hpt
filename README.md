# Higgs pT

## Install

First, create a virtual environment (`micromamba` is recommended):

```bash
# Download the micromamba setup script (change if needed for your machine https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
# Install: (the micromamba directory can end up taking O(1-10GB) so make sure the directory you're using allows that quota)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# You may need to restart your shell
micromamba create -n hh4b python=3.10 -c conda-forge
micromamba activate hh4b
```

### Installing package

**Remember to install this in your mamba environment**.

```bash
# Clone the repository
git clone https://github.com/cmantill/hpt.git
cd hpt
# Perform an editable installation
pip install -e .
```

## Run locally

e.g. for a test
```
python -u -W ignore src/run.py --year 2023  --starti 0 --endi 1 --samples VJets --subsamples Zto2Q-4Jets_HT-400to600 --processor ptSkimmer --nano-version v12
```

## Submit jobs

e.g. for Z+Jets
```
```

