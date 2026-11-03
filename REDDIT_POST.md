[P] I built a synthetic dataset with known ground truth and it found three bugs in my own models

**The problem**

Every data science project ends with a number. R² 0.7, AUC 0.85, recall@10 0.2. The number looks plausible, the notebook runs, and nobody asks the uncomfortable question: did the model learn the mechanism that generates the data, or did it memorise noise?

Cross-validation does not answer this. It tells you the model generalises to similar data. It cannot tell you whether the signal it learned is the right one.

**The setup**

I was building a B2B distribution analytics project - price elasticity, churn, SKU recommendation - on top of a public synthetic dataset (Atacado Atlas, 22 tables, 52k orders). The results were bad:

- Elasticity: R² 0.004
- Churn: AUC 0.537 (worse than a coin flip)
- Recommendation: recall@10 0.208

Is my code wrong, or is there nothing in the data? The Atlas cannot answer. It records no true elasticity, contains no churners (all 480 customers buy in all 9 quarters), and has no documented latent structure.

**Building the instrument**

So I generated a dataset where the answer is known. Same operational shape (12 categories, 5 price tables, 730 days, BRL), but with three behavioural signals injected from documented parameters:

1. Price elasticity: qty = k × (price/full_price)^(-ε), ε per SKU between 0.8 and 2.0, with price driven causally by real CEPEA commodity indices
2. Churn: latent health decline that also drives order rate and basket width, so the precursors are observable before the label
3. Affinity: 8-dimensional latent vectors per customer and SKU

Scale: 5,000 customers, 500 SKUs, 475k orders, 2.1M line items. Generated in about a minute.

**What it found**

Bug 1 - elasticity specification. My original model was `log_qty ~ log_preco * C(categoria)`. The problem: `log_preco` mixes three sources of variation (price table assigned to the customer, monthly list price change, per-order promotion). Only the last is demand response. Within-SKU × month fixed effects isolates it. R² went from 0.106 to 0.700, and the correlation with the true ε is 0.995.

Bug 2 - recommendation evaluation. My protocol hid "the last item purchased" per user. But "last by volume" is always the smallest purchase, which the model never ranks highly. Leave-one-out over 500 sampled users took recall@10 from 0.015 to 0.244.

Bug 3 - churn leakage. I reported AUC 0.988 against the ground truth. It was temporal leakage: recency was computed at the end of the dataset, so for any customer who had already churned it grew without bound. Temporal split (features up to month t-6, label after t) gave 0.538 - and revealed that the generator's hazard was drawn independently of behaviour. There was no signal to learn. Rewriting the generator to tie hazard to an observable decline brought the honest number to 0.675.

**The lesson I keep repeating to myself**

A number that looks too good is an invitation to verify, not a result to publish. I had 0.988 in three documents before checking it.

**Links**

- Dataset + generator: https://github.com/vinidias/GiroAI-SYNTH
- Pipeline: https://github.com/vinidias/GiroAI-B2B-SKU-recommendation-price-elasticity-churn-and-segmentation
- Visual report: https://vinidias.github.io/GiroAI-SYNTH/report.html

Happy to answer questions about the generator design, especially the tension between making a signal recoverable (useful for auditing) and realistic (harder to generate).
