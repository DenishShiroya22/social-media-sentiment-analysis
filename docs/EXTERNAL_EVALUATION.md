# External brand-domain evaluation plan

A future application evaluation must use a newly collected, manually labeled set of real public brand or product comments. It must remain separate from TweetEval TRAIN, VALIDATION, and the completed official TEST benchmark.

## Proposed set

- Collect approximately 300–500 public comments through the application connectors.
- Predefine a small set of brands, products, sources, dates, languages, and query rules.
- Sample across queries and time windows rather than taking only highly ranked posts.
- Remove exact duplicates and document each exclusion without selecting examples based on model output.
- Keep source, source ID, collection date, public text, query, and only the minimal metadata needed for auditing.
- Omit author identity unless a reviewed use case requires it.
- Store the working data under data/external_eval, which Git ignores.

## Labeling

Create written negative, neutral, and positive guidelines with examples independent of model predictions. Use at least two human annotators where practical. Resolve disagreements through adjudication and report both initial agreement and final class counts. Mark genuinely ambiguous or unusable records before evaluation according to predefined rules.

Annotators must not see the model’s label, probability, TweetEval TEST result, or experimental model comparison while assigning labels. Do not manufacture labels or scores.

## Evaluation protocol

Freeze the collected IDs, labels, preprocessing contract, model artifact SHA-256, metrics, and evaluation script before predicting. Check that no item overlaps the benchmark by exact normalized text and source identifiers. Evaluate the existing application model once with:

- accuracy;
- macro precision, recall, and F1;
- negative, neutral, and positive precision, recall, and F1;
- confusion matrix;
- class support;
- a paired uncertainty analysis only when comparing a separately frozen candidate.

Document sampling limitations, annotation uncertainty, domain coverage, and missing context. Do not claim population-level performance from a small convenience sample.

## Privacy and publication

Follow source terms and retention requirements. Avoid private or deleted content, emails, phone numbers, private messages, and hidden profile data. Prefer publishing aggregate results and labeling instructions; do not commit collected text to the public repository without a separate rights and privacy review.

This future work must not partition, relabel, or rerun the official TweetEval TEST split.
