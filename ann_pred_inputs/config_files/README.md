# Config files for running experiments
They are organized by network.
- `effective-diffusion.yaml`: includes `alpha` values to test for RL
- `params-testing.yaml`: includes parameter values to test for each method

## stringv11
- STRING network using all string channels with a 'combined' cutoff of 400 (medium quality)

## stringv11-exp
- STRING network using the 'experiment' string channels with a 'combined' cutoff of 900 (very high quality)

## biogrid  
- All PPIs from the BioGRID database (excluding genetic interactions)

## biogrid-y2h
- Only direct PPIs from yeast 2-hybrid (Y2H) screens 

## HI-union  
- The high-quality “HI-union” network published by Luck et al.
  - Luck K, Kim DK, Lambourne L, et al.  A reference map of the human binary protein interactome. Nature. 2020;580(7803):402–8.

## match-string-size-to-nets
- Reduce the size of the string network until it matches the other networks

