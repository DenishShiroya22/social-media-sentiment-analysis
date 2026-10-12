# External sentiment labeling guide

## Purpose and blinding

Label the sentiment expressed in each public brand/product comment. Use only `negative`, `neutral`, or `positive`. Annotators must not see model predictions, confidence scores, benchmark results, or experimental model comparisons. The CSV `label` column is intentionally blank when exported.

## Labels

- **negative**: clear dissatisfaction, criticism, complaint, disappointment, rejection, or negative experience.
- **neutral**: factual or informational content, a genuine question without clear polarity, unclear/ambiguous sentiment, or no clear approval or disapproval.
- **positive**: clear praise, satisfaction, enthusiasm, approval, recommendation, or positive experience.

## Edge cases

- **Sarcasm:** label the intended sentiment when it is reasonably clear; otherwise choose neutral and set `review_flag`.
- **Mixed sentiment:** choose the dominant expressed sentiment. If neither side dominates, choose neutral and flag for review. Do not create a fourth label.
- **Comparisons:** label sentiment toward the queried product. “A is better than B” is positive for A and negative for B when the target is clear.
- **Questions:** factual questions are neutral; rhetorical questions carrying clear criticism or praise follow that polarity.
- **Factual statements:** neutral unless wording clearly expresses approval or dissatisfaction.
- **Quoted sentiment:** label the writer's own stance, not an unendorsed quotation. Flag unclear attribution.
- **Emojis:** treat emojis as context; do not infer polarity from one symbol when the full message conflicts.
- **Negation:** apply the negation to the sentiment phrase, such as “not good” → negative.
- **Profanity:** profanity alone is not automatically negative; use the target and expressed stance.
- **Irrelevant comments:** use neutral and add `irrelevant` to `review_flag` or notes for later documented exclusion.
- **Language uncertainty:** rows marked `language_review` need an English-language eligibility decision. If the annotator can understand the row reliably, label it and remove `language_review`. If translation would be required, do not assign neutral as a substitute for sentiment; preserve the row in a private exclusion file and omit it from the evaluation CSV.
- **Missing context:** use neutral and flag when a reply cannot be interpreted reliably without unavailable context.

## Workflow

1. Open the UTF-8 CSV in a spreadsheet tool.
2. Fill every `label` cell with exactly `negative`, `neutral`, or `positive`.
3. Use `review_flag` and `notes` for ambiguity; do not edit IDs or text.
4. Save a separate file named `external_eval_labeled.csv`.
5. Run the label validator before evaluation.

A second annotator is optional. If used, work in separate copies, measure agreement outside the blind labeling stage, adjudicate disagreements, and place the final result in the single `label` column.
