# Git Setup and Push Instructions

Since automated git operations require special permissions, please run these commands manually:

## 1. Initialize Git Repository

```bash
cd /Users/keshavkrishnan/Oil_Project
git init
git config user.name "Keshav Krishnan"
git config user.email "keshav-krishnan@outlook.com"
```

## 2. Add Remote Repository

```bash
git remote add origin git@github.com:keshavkrishnan08/TFTEnergyTrading.git
```

## 3. Stage Essential Files

```bash
# Core source code
git add src/

# Main experiment scripts
git add main_tft_v8_sliding.py
git add main_advanced.py
git add main_meta_learning.py

# Statistical tests
git add monte_carlo_comprehensive.py
git add monte_carlo_simulation.py
git add run_monte_carlo_baselines.py

# Documentation
git add README.md
git add EXPERIMENTAL_SETUP.md
git add .gitignore

# Paper
git add ieee_tnnls_paper_enhanced_v3.tex
git add references.bib

# Key experiment results (if needed)
git add experiments/tft_v8_sliding/
git add experiments/sliding_window_v7/
git add experiments/baselines_comprehensive/
```

## 4. Commit and Push

```bash
git commit -m "Initial commit: TFT-VSN commodity trading system

- Complete TFT-VSN implementation with Variable Selection Networks
- Meta-learner for trade selection and position sizing
- Monte Carlo bootstrap simulations (50K samples)
- Statistical validation tests (Jobson-Korkie, Diebold-Mariano, White's Reality Check)
- Comprehensive experimental setup documentation
- IEEE TNNLS paper and bibliography"

git branch -M main
git push -u origin main
```

## 5. Verify Push

```bash
git log --oneline
git remote -v
```

## Files Included

The `.gitignore` file has been configured to exclude:
- Large data files (*.csv, except key experiment results)
- Model checkpoints (*.pt, *.pth)
- Log files (*.log)
- Old experiment folders
- Development files and notebooks
- Paper drafts (except final version)

Only essential files for reproduction are included in the repository.

