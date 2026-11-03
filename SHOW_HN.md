Show HN: GIRO-SYNTH – a synthetic dataset with known ground truth to audit ML pipelines

I kept running into the same problem: a model reports an AUC or R², the number looks plausible, and there is no way to tell whether it learned the mechanism or just memorised the past. Cross-validation does not answer that question - it only says whether the model generalises to similar data.

I was working on a B2B distribution project (price elasticity, churn, SKU recommendation) on top of a public synthetic dataset. The results were weak: elasticity R² = 0.004, churn AUC = 0.537. The obvious question was "is my code wrong or is there nothing in the data?" The dataset could not answer it.

So I built one that can. GIRO-SYNTH generates 475k orders across 5,000 customers and 500 SKUs, where three behavioural signals come from documented parameters:
- price elasticity per SKU (0.8-2.0), driven by a causal chain from real CEPEA commodity prices to SKU cost to price to demand
- churn from a latent health decline that also drives order rate and basket width, so the precursors precede the label
- 8-dimensional latent customer-SKU affinity

Ground truth is published alongside the data, so anyone can measure whether an estimator recovers the true parameter.

The useful part was what it did to my own pipeline:

1. Elasticity: a pooled OLS mixed three sources of price variation (price table, monthly list, per-order promotion). Only the last is demand response. Within-SKU × month fixed effects took R² from 0.106 to 0.700, Pearson 0.995 against the true epsilon.

2. Recommendation: hiding "the last item" per user holds out the smallest purchase, which the model never ranks. Leave-one-out over 500 users took recall@10 from 0.015 to 0.244.

3. Churn: I reported AUC 0.988. It was temporal leakage - recency computed at the end of the dataset grows without bound for anyone who already churned. A temporal split gave 0.538, and showed the generator's hazard was independent of behaviour, so there was no signal at all. Fixing the generator brought the honest number to 0.675.

That last one is the reason I am posting this. I had 0.988 in three documents before I checked it. A number that looks too good is an invitation to verify.

Code and dataset (MIT + CC BY 4.0):
- https://github.com/vinidias/GiroAI-SYNTH
- https://github.com/vinidias/GiroAI-B2B-SKU-recommendation-price-elasticity-churn-and-segmentation
- visual report: https://vinidias.github.io/GiroAI-SYNTH/report.html

Happy to discuss the design, especially the trade-off between realism and identifiability - a synthetic dataset can make a signal recoverable by construction, which is useful for auditing but obviously easier than the real problem.
