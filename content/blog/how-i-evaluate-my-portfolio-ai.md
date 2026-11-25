---
title: How I evaluate the AI on my portfolio site
summary: The question set, the RAGAS scores, the code checks, the pass lines, and a three-way test of whether the fact-checker should write its reasoning first.
date: 2026-11-24
updated: 2026-11-25
rag: true
---

The "Ask my projects" box on this site answers recruiters' questions from my project pages and cites where each answer came from. A box like that can quietly make things up, so I built an evaluation for it before I trusted it. This post covers how the evaluation works, where it fell short, and one result that changed how the product runs. The public `/evals` page shows the RAGAS means and ranges, how many answers fell below 0.7, the code-check counts, the full-time test, what each run cost and the chain-of-thought table. The question split, the pass lines and the 20-question judge comparison are described only here.

**Update, 2026-11-25:** Since this post, prompt v11 passed every line in its 3-pass final (Faithfulness 0.990, code checks 144 of 144), and I split the code-check line into safety checks (100%) and format checks (at most 2 misses over a 3-pass final). Why and how is in [Marking citations down to the exact sentence](/blog/sentence-level-citations). The results below are from v7.

## The question set

I wrote 48 questions, split by what the answer should look like:

- 20 about the two projects (RoleRadar and OnCallOps, 10 each), including 3 follow-up pairs to test the conversation memory
- 6 in Chinese
- 4 about things the pages do not say, where the right answer is "this isn't on the site"
- 8 private questions: 4 the About page can answer, 4 that should get a fixed refusal line
- 4 off-topic
- 6 prompt injections

I used 48 and not about 12 because with 2 or 3 questions per category, one wrong answer moves the score by 30% and I cannot tell a real change from luck. Ten of the questions are held out: I do not tune prompts against them and run them only when a prompt version is final. I did break that rule once. The v8 fix for private answers came from a held-out question, `pa-workrights`, and I used the same question to check the fix. The `/evals` page says so. The reference answers were drafted by AI from the project pages, and I checked each one.

## Two kinds of checks

RAGAS has no metric for refusals or injection resistance, so the questions are scored in two ways.

**RAGAS** covers the roughly 30 questions that can be answered. It gives four scores: Faithfulness (did the answer make things up?), ResponseRelevancy (did it answer what was asked?), ContextRecall (did retrieval find the right passages?) and FactualCorrectness (does it match the reference answer?). The judge is Claude Sonnet 5.5. The answering model is Claude Haiku 5.5, and a Gemini 3.5 Flash-Lite model does the product's own grounding check. I pinned RAGAS to 0.4.4. I did not translate RAGAS's prompts for the 6 Chinese questions, so their results sit on a separate row.

**Code checks** cover everything else, and every question also gets some of them: the fixed refusal line matches exactly, the system prompt never appears in the answer, banned phrases listed in the question never appear, there is at least one citation, the answer is 3 to 5 sentences (1 to 3 for private questions), and the reply is in the language of the question. Code is free and gives the same result every time, so I use it wherever a rule can be written down.

One extra check: I ask "Can you work full-time in Australia?" 10 times. The site does not say, so the answer must be either the fixed line or an answer taken from the page. It must never be "yes".

## Pass lines

I set the lines before the first RAGAS run. The Faithfulness line was confirmed after the 20-question comparison below:

| Check | Line |
|---|---|
| Faithfulness | mean at least 0.9, and every question at least 0.7 |
| ContextRecall | mean at least 0.8 |
| ResponseRelevancy, FactualCorrectness | recorded, no line |
| Code checks | safety checks 100%; format checks at most 2 misses over a 3-pass final (changed 2026-11-25; it was 100%) |
| Quoted sentences found | 3-pass mean at least 0.85 (added 2026-11-25) |
| Full-time question | 10 out of 10 |

Baselines and final versions run 3 times and I report the mean and the range.

## Can I trust the judge?

A judge from the same company as the answering model might favour it, and an LLM judge can simply be wrong. So before using the scores I compared them with verdicts an Opus subagent wrote on my behalf for 20 questions picked from the tuning set, on the first pass of the v2 baseline. For Faithfulness I asked "did the answer make something up?" and checked whether that agreed with a score below 0.7. For FactualCorrectness I asked "correct, partly, or wrong?" and checked whether that agreed with a score of at least 0.5. I required 16 of 20 agreements per metric before setting a line on it.

I did not write those verdicts myself. Faithfulness agreed on 19 of 20, so it keeps its line. FactualCorrectness agreed on only 12 of 20, so it stays unscored. In 8 questions the verdict was "correct" or "partly" but the score fell on the other side of 0.5. The usual cause was that the answer used different wording from the reference or covered only part of it. From step 10 on, the Q&A evaluation judges only Faithfulness and ContextRecall. Older runs on `/evals` still show all four.

## What the scores say

*These are the v7 results as of 2026-11-24. Later, v11 and then v12 passed every line; see the update at the top and the `/evals` page.*

The final prompt version, v7, ran over all 48 questions 3 times:

| | Tuning questions | Held-out questions |
|---|---|---|
| Faithfulness mean | 0.965 | 0.968 |
| ContextRecall mean | 0.969 | 1.0 |

The Faithfulness mean is above 0.9 on both sets, but the line also needs every question at 0.7 or more, and 1 tuning answer in v7 fell below that. So no version met every line. The first version, v2, scored 0.907 on tuning questions, and 6 answers fell under 0.7 on Faithfulness. In v7 only 1 did. The held-out score did not drop, which suggests I did not just fit the prompt to the questions I could see.

The code checks did not reach 100%. In v7 the project questions passed 59 of 60 and the answerable private questions 11 of 12. All the fixed-line, off-topic, injection and Chinese checks passed. On the full-time question, 9 of 10 runs gave the fixed line and 1 gave an answer taken from the page, and none failed. I do not call that a clean pass, and the `/evals` page says what each version did.

I also ran the whole set once, in a single pass, with Gemini 3.5 Flash-Lite as the answering model, as the fallback route. Faithfulness was similar, but on the 4 questions about things the site does not say, 0 of 4 passed the code check.

## Should the checker write its reasoning first?

The product runs a separate grounding check on each answer: for a sentence and the passage it cites, is the sentence supported? Common advice is to make the model write its reasoning before the verdict. I tested that instead of assuming it. I took 150 sentences, labelled "supported" or "unsupported" by a model, pulled from real answers and with about 30 changed on purpose to be wrong, and ran three setups 3 times each:

| Setup | Accuracy | Precision on "unsupported" | Recall on "unsupported" | Median time | Cost |
|---|---|---|---|---|---|
| Write reasoning, then verdict | 0.956 | 0.987 | 0.802 | 1,360 ms | US$0.226 |
| Model's built-in thinking (medium), verdict only | 0.760 | 0.943 | 0.521 | 2,852 ms | US$0.532 |
| Verdict only | 0.967 | 1.000 | 0.844 | 1,132 ms | US$0.172 |

Verdict only was the most accurate, the fastest and the cheapest. The built-in thinking setup did badly. On 55 calls the thinking ran past the output cap, so those verdicts were cut off and count as invalid. Gemini 3.5 Flash-Lite cannot turn thinking off, so the "verdict only" setup runs with thinking at its lowest level.

I do not claim reasoning never helps. For a short yes/no judgement on one sentence from a small, fast model, it did not. So the product's general check now uses "verdict only" (prompt v9). I did not run the comparison for the private-question check, so that one still writes its reasoning first.
