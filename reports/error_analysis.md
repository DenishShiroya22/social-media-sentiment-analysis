# VALIDATION error analysis

Errors come from the validation-selected candidate only. Samples are ordered, small and illustrative, not representative estimates.
TEST is locked. No TEST predictions or performance were generated.

## Confusion counts

Rows are true labels; columns are predictions in negative, neutral, positive order.

[[141, 135, 36], [59, 667, 143], [25, 192, 602]]

Total errors: 590 of 2000 validation rows.

## negative to neutral: 135

- Row 2: when girls become bandwagon fans of the packers because of harry. do y'all even know who aaron rodgers is? or what a 1st down is?
- Row 10: so the thing next thursday isn't free, you'd have to pay $15 to get in since you don't go to umbc :/ and it ends at 11:30"

## negative to positive: 36

- Row 90: "judging by the traffic and complaining, i think i might be best setting for foo fighters' milton keynes gig tomorrow right now"
- Row 212: i think i may have spent too much on carmelo anthony. i only hope that his thick-framed black glasses will console me in the post-season.

## neutral to negative: 59

- Row 40: look steelers fans i know you may be upset about suisham missing that kick. just know that i heard a guy named billy cundiff is available.
- Row 85: ok big diff lmao my parents were boaters they didn't know a lot abt islam when they came. my oldest sis wore it in 1st

## neutral to positive: 143

- Row 23: in this second time i've watched ant-man and this time i was the only one that stayed for the 2nd after credits scene
- Row 27: every time i hear alright by kendrick i think it's j cole's black friday

## positive to negative: 25

- Row 148: "rahul gandhi is a ppl's leader. modimafia may try as hard, they will not be able defeat him simply becuz ppl's voice just cannot be curbed."
- Row 197: "\"""" : cupid doesn\u2019t lie but you won\u2019t know unless you give it a try\"""" wwooww.may pinaghuhugutan!"

## positive to neutral: 192

- Row 6: us 1st lady michelle obama speaking at the 2015 beating the odds summit to over 130 college-bound students at the pentagon office. >>
- Row 16: tom brady is locked for thursday. let the season begin! repeatseason

## Descriptive cues in misclassified validation text

- Explicit negation token: 102 errors.
- Exclamation or question mark: 178 errors.
- Five words or fewer: 0 errors.

Illustrative reading: negative-to-neutral examples can express dissatisfaction through price or access details rather than a simple negative adjective. Positive-to-neutral examples can depend on future expectations or event context. Neutral-to-positive examples may contain named events without an explicit opinion. These are hypotheses from a few displayed posts, not measured error causes.
These counts do not establish causes. Sarcasm, mixed sentiment, named entities, tense and missing conversation context require manual review. The displayed examples should guide focused future hypotheses, then be tested on validation without unlocking TEST.
