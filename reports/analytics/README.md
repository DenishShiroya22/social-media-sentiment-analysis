# External aggregate analytics

These reports summarize frozen-model predictions over the private 331-row human-labeled external dataset. No row-level comment text, representative comments, usernames, or credentials are stored here.

Current aggregate prediction counts are 66 negative, 169 neutral, and 96 positive. Average model confidence is 0.6674 and median confidence is 0.6724; these scores are not guaranteed to be calibrated probabilities.

`external_analytics.json` contains overall and per-query sentiment, confidence summaries, and daily and weekly trends. `external_topics.json` contains descriptive TF-IDF terms and configurable product-theme counts. The current matched themes include pricing, delivery, battery, design, customer support, performance, bugs, and availability. Topic labels and keyword matches are imperfect descriptive aids.

Regenerate the aggregate reports from the ignored local labeled file with:

```powershell
.\.venv\Scripts\python.exe -m src.analytics.report --input data/external_eval/labels/external_eval_labeled.csv
```
