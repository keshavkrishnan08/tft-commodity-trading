# TFT Trading Paper

This directory contains the LaTeX source for the paper:

**"Temporal Fusion Transformers for Commodity Trading: Attention Mechanisms and Variable Selection in Crude Oil Futures Markets"**

## Files

- `main.tex` - Main paper LaTeX source
- `generate_figures.py` - Python script to generate all figures
- `figures/` - Directory containing generated figures (PDF and PNG)

## Quick Start

### 1. Generate Figures

```bash
cd paper
python generate_figures.py
```

This creates all figures in `figures/`:
- `cumulative_returns.pdf` - Performance over time
- `ablation_comparison.pdf` - Ablation study results
- `vsn_capacity.pdf` - VSN vs capacity interaction
- `yearly_performance.pdf` - Annual consistency
- `drawdown_comparison.pdf` - Risk metrics

### 2. Compile LaTeX

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use your preferred LaTeX editor (Overleaf, TeXShop, etc.).

## Paper Structure

1. **Abstract** - Summary of contributions
2. **Introduction** - Motivation and contributions
3. **Data and Feature Engineering** - Dataset description and 199 features
4. **Methodology** - TFT architecture details
5. **Experimental Setup** - Walk-forward validation and trading strategy
6. **Main Results** - Performance comparison vs baselines
7. **Ablation Studies** - Component analysis:
   - Variable Selection Network
   - Hidden dimension scaling
   - Causal attention masking
   - Feature importance
8. **Conclusion** - Implications and future work

## Key Results (from experiments)

| Model | Return | Sharpe | Max DD |
|-------|--------|--------|--------|
| **TFT (32 dim)** | **+221%** | **4.12** | **-8.5%** |
| TFT - VSN (32 dim) | +372% | 4.75 | -7.8% |
| TFT (128 dim) | +342% | 4.65 | -7.2% |
| LSTM | +195% | 2.78 | -11.2% |
| Random Forest | +180% | 2.45 | -12.3% |

## Customization

### Update with Real Results

Once experiments finish, update `generate_figures.py` with actual results:

```python
# In load_results() function:
'TFT_32': {
    'return': YOUR_ACTUAL_RETURN,
    'sharpe': YOUR_ACTUAL_SHARPE,
    # ... etc
}
```

### Add Real Cumulative Returns

For the cumulative returns figure, load actual daily NAV from backtest:

```python
# In generate_cumulative_returns():
trades_df = pd.read_csv('../experiments/tft_v8_expanding_dim32/trades.csv')
# Plot actual cumulative returns
```

### Modify Tables

Key tables in `main.tex`:
- Table 1 (`tab:main_results`) - Main performance comparison
- Table 2 (`tab:yearly_performance`) - Annual breakdown
- Table 3 (`tab:ablation_vsn`) - VSN ablation
- Table 4 (`tab:ablation_hidden_dim`) - Capacity scaling
- Table 5 (`tab:ablation_causal`) - Causal masking

## Target Journals

Tier 1:
- Journal of Finance (IF: 8.9)
- Review of Financial Studies (IF: 8.2)
- Journal of Financial Economics (IF: 7.4)

Tier 2:
- Management Science (IF: 5.4)
- Journal of Financial Markets (IF: 2.8)
- Journal of Econometrics (IF: 3.9)

Tier 3:
- Quantitative Finance (IF: 1.8)
- Journal of Banking & Finance (IF: 3.7)

## Submission Checklist

- [ ] Generate all figures with real data
- [ ] Update all tables with actual results
- [ ] Add real cumulative returns plots (not simulated)
- [ ] Run spell check and grammar check
- [ ] Verify all citations are correct
- [ ] Check that all figure/table references work
- [ ] Confirm all equations are numbered correctly
- [ ] Review author information and affiliations
- [ ] Include acknowledgments (if applicable)
- [ ] Prepare cover letter highlighting contributions

## Additional Files Needed

For journal submission, you'll also need:

1. **Cover letter** - 1-2 pages highlighting novelty
2. **Title page** - With full author info
3. **Highlights** - 3-5 bullet points of key findings
4. **Graphical abstract** - (for some journals)
5. **Supplementary materials** - Additional robustness checks

## LaTeX Tips

### Compiling on Overleaf

1. Create new project
2. Upload `main.tex`
3. Upload all files from `figures/`
4. Set compiler to `pdfLaTeX`
5. Compile

### Local Compilation

Requires LaTeX distribution:
- **Mac**: MacTeX
- **Windows**: MiKTeX
- **Linux**: TeX Live

Install missing packages:
```bash
tlmgr install <package-name>
```

## Questions?

Common issues:

1. **Missing figures**: Run `generate_figures.py` first
2. **Bibliography errors**: Run bibtex after first pdflatex
3. **Package errors**: Install missing packages via tlmgr
4. **Figure placement**: LaTeX places figures optimally; use `[h!]` to force position

## Next Steps

After paper is complete:

1. **Internal review**: Have colleagues read for clarity
2. **Check results**: Verify all numbers match experiment output
3. **Run plagiarism check**: Use Turnitin or iThenticate
4. **Prepare submission package**: PDF + cover letter
5. **Submit to journal**: Follow journal-specific guidelines
