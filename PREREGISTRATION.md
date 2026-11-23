# Pre-registration: Meta's GenRecs tokenization (semantic IDs) on MovieLens

Written before any model was trained. Committed to git so the timestamp is checkable.

## Source
Meta Engineering, "GenRecs Tokenization" (Hu, Johnson, Aggarwal, Rangadurai, Manickavelu, Zhang, Oct 2026).
The public method it builds on: semantic IDs with prefix n-grams (Singh et al., "Better Generalization with Semantic IDs", arXiv 2306.08121).
- C1. Items represented by tokens from a stable semantic vocabulary (content embedding, residual decomposition, codebooks, prefix n-grams) let new and tail items borrow what the model learned about similar items.
- C2. Combining the content signal with engagement-learned representations makes better tokens.
- C3. Across 8 tokenizer versions, three intrinsic metrics (cluster purity, distribution balance, codebook utilization) "correlated strongly" with downstream prediction accuracy.

## Setup
- Data: real MovieLens ratings (about 890,000). Target: liked (rating >= 4) vs not, among rated pairs. A stand-in for click/conversion prediction.
- Content: CLIP (ViT-B/32, laion2b) text embedding of "title, genres" for every movie.
- Engagement embedding: item factors from a truncated SVD of the liked matrix on training ratings only.
- Cold items: 20% of movies (stratified by popularity) never appear in training. Warm test: 10% of ratings on the other movies, held out at random.
- Tokenizers: residual k-means, 3 levels; codebook sizes and inputs varied to make 12 versions (content-only, engagement-aligned, random projection baseline, different codebook sizes).
- Engagement-aligned tokens for cold items: a ridge map from content to engagement space, fitted on warm items, then tokenized.
- Downstream model: the same small PyTorch model for every item representation (user embedding dot item representation plus biases), same epochs and seed. Item representations: ID embedding; dense content projection; semantic IDs (sum of prefix n-gram embeddings); ID + semantic IDs.
- Metric: AUC, with a user-level bootstrap for intervals.

## Hypotheses and decision rules
H1 (C1). On cold items, semantic IDs beat the ID model (which falls back to a shared unknown embedding) and beat the dense content model.
  PASS if AUC(SID) > AUC(dense content) on cold items with bootstrap 95% CI of the difference above 0.
H2. On warm items, semantic IDs alone are within 0.005 AUC of the ID model.
H3 (C2). Engagement-aligned tokens beat content-only tokens on cold items. PASS if CI of the difference above 0.
H4 (C3). Across 12 tokenizer versions, at least one of the three intrinsic metrics has Spearman |rho| >= 0.6 with cold-item AUC, p < 0.05.
H5 (the attempt to break it). Token sharing has a cost for head items: on the 10% most-rated warm items, ID + SID beats SID alone. PASS if CI of the difference above 0.
