# Responsible data collection policy

## API-first rule

External evaluation data must come from platform-supported public APIs where practical. This repository supports read-only Reddit collection through PRAW and public YouTube video/comment retrieval through YouTube Data API v3. It does not scrape Reddit or YouTube HTML.

The default configuration is deliberately bounded. Reddit limits submissions and top-level comments per submission. YouTube limits matching videos and top-level comments per video. Operators must keep limits conservative, respect platform quotas and terms, and use a descriptive Reddit user agent.

## Credential handling

Credentials come from process environment variables only:

- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`
- `YOUTUBE_API_KEY`

Never place values in source, configuration, logs, manifests, notebooks, or committed `.env` files. `.env.example` contains empty placeholders only. The YouTube connector uses read-only `search.list` and `commentThreads.list` requests. Google documents API-key authentication for public requests and a `commentsDisabled` response when a video does not expose comments.

## Privacy minimization

Collect public text, stable platform ID, UTC creation time, query, public URL, minimal engagement counts, and a thread/video context ID. Author names are disabled and stripped before external-evaluation storage. Do not collect emails, phone numbers, private messages, hidden profiles, private account data, or sensitive-trait inferences.

Reports may contain aggregate counts and metrics. They must not publish raw comments, usernames, tokens, or row-level predictions. Real raw, prepared, and labeling files remain local and Git-ignored. Redistribution requires a separate review of platform terms, rights, privacy, retention, and deletion obligations.

## Scraping fallback

No scraping adapter is implemented. A future adapter is permitted only after documenting that no practical API exists and collection is allowed by the source's terms, policies, and `robots.txt`. It must use public pages without login, CAPTCHA bypass, anti-bot evasion, rotating proxies, fingerprint spoofing, or hidden APIs. It must identify itself appropriately, bound requests, rate-limit, back off, and remain isolated under `src/social/scraping/`.

## Failures and retention

Missing or invalid credentials, quota exhaustion, network failures, disabled comments, and malformed responses produce explicit outcomes. The workflow never fabricates fallback rows. Remove locally retained content when it is no longer needed or when source obligations require deletion.

Official references:

- [YouTube Data API reference](https://developers.google.com/youtube/v3/docs)
- [YouTube `commentThreads.list`](https://developers.google.com/youtube/v3/docs/commentThreads/list)
- [PRAW read-only quick start](https://praw.readthedocs.io/en/latest/getting_started/quick_start.html)
