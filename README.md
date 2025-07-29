# Project Setup Guide

## Quick Setup Steps

### 1. Install Environment
```bash
conda env create -f environment.yml
conda activate spaceship
```

### 2. Get Project Files
```bash
git clone <repository-url>
cd FinalProject
```

### 3. Pull Data
```bash
dvc pull
```

## Working with Data

### Pull Latest Data
```bash
dvc pull
```

### Push New Data
```bash
# Add your data file
dvc add data

# Commit to git
git add data/your-file.csv.dvc .gitignore
git commit -m "Add new data"

# Push data to storage
dvc push

# Push code changes
git push
```

### Push Code Changes
```bash
git add .
git commit -m "your message"
git push
```

That's it! 🚀