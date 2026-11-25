---
title: Picking a model for each step of a job-match pipeline
summary: I tried three models at each of four steps, compared quality, cost and latency, and let a cost line decide ties. Here is the table and what I picked.
date: 2026-11-24
updated: 2026-11-24
rag: true
---

The job-match feature on this site takes a job ad and tells you which parts of my project pages speak to each requirement. It is a small pipeline with four steps, and each step has its own model. My first version used one setup for everything and missed too many requirements. For round two I stopped treating the model as one choice and picked it step by step.

This post covers what I tried, what I measured, and how I made the choices.

## The four steps

1. **Read the ad.** A model pulls out the requirements, up to a limit.
2. **Match.** For each requirement, a model looks at the retrieved sections of my project pages and says whether the requirement is covered, and where.
3. **Check.** A second model looks at each requirement and section pair and says whether the section really talks about that requirement. Sections that fail are dropped.
4. **Write.** A model writes the short guide the visitor reads, using only the sections that survived the check.

The first round showed that the misses were not all in one place. Of the 108 times a requirement's section was missed, 48 were lost at reading, 16 at retrieval, 24 at matching and 20 at the check. Different steps had different problems, so one model swap could not fix them all.

## What I tried

I tried three options per step and replayed each one on the stored intermediate output of one earlier full run. That meant I only re-ran the step under test, on 26 jobs (21 to tune on, 5 held out that I did not use for choosing).

| Step | Options |
|---|---|
| Read | Haiku 5.5 thinking off; Haiku 5.5 low thinking; Sonnet 5.5 |
| Match | Haiku 5.5 thinking off; Haiku 5.5 low thinking; GPT-6 Luna |
| Check | Gemini 3.5 Flash-Lite minimal; Flash-Lite low; GPT-6 Luna |
| Write | Haiku 5.5 thinking off; Haiku 5.5 low thinking; Sonnet 5.5 |

For reading I also tried a different limit: at most 12 requirements instead of 8. The match and check steps were deliberately kept on different providers.

## The comparison

Quality is the score on the 21 tuning jobs, and each step has its own score, so compare rows within a step, not across steps. Read is the share of the expected requirements that were read out. Match is the share of expected sections the model picked from those retrieved. Check is the share of requirement and section pairs judged correctly. Write is faithfulness: whether each claim is supported by the section it cites. Cost and latency are means per job ad over all 26 replay jobs (tuning plus held-out), for that step only, rounded; for the match step, latency is the slowest of the parallel calls, not their sum.

| Step | Option | Quality | Cost | Latency |
|---|---|---|---|---|
| Read | Haiku 5.5, off | 0.8987 | $0.0005 | 2.5 s |
| Read | Haiku 5.5, low | 0.8734 | $0.0005 | 2.7 s |
| Read | Sonnet 5.5 | 0.8987 | $0.0093 | 3.9 s |
| Read | Haiku 5.5, off, up to 12 | 0.943 | $0.0005 | 2.6 s |
| Match | Haiku 5.5, off | 0.9583 | $0.0040 | 2.5 s |
| Match | Haiku 5.5, low | 0.9688 | $0.0040 | 2.6 s |
| Match | GPT-6 Luna, low | 0.8854 | $0.0028 | 5.0 s |
| Check | Flash-Lite, minimal | 0.8339 | $0.0028 | 2.4 s |
| Check | Flash-Lite, low | 0.8407 | $0.0028 | 1.5 s |
| Check | GPT-6 Luna | 0.8136 | $0.0007 | 2.0 s |
| Write | Haiku 5.5, off | 0.9744 | $0.0010 | 2.5 s |
| Write | Haiku 5.5, low | 0.9732 | $0.0016 | 7.0 s |
| Write | Sonnet 5.5 | 0.9909 | $0.0200 | 4.6 s |

The Evals page lists the same options with held-out scores and totals per match, which include the other three steps, so its per-match numbers differ from the per-step ones here.

## How I chose

I wrote the rule before I saw the numbers, so I could not bend it afterwards:

- Pick the best score on the tuning jobs for each step.
- If two options are within 0.02 of each other, call it a tie and take the cheaper one.
- Do not use the same provider for the match step and the check step.
- An option over the per-match cost line cannot be chosen.
- Held-out jobs are reported but never used to choose.

Applying it:

- **Read.** Sonnet matched Haiku (both 0.8987) at about twenty times the cost, so Haiku. Low thinking was no better. The one real difference was the limit: allowing up to 12 requirements scored 0.943 against 0.8987 at the same cost. Reading more requirements also means more work downstream (about 15% more per run), so I made it a deliberate decision, and 12 became the limit.
- **Match.** Haiku off (0.9583) and Haiku low (0.9688) are inside the tie band, so I took the marginally cheaper one and kept thinking off. Luna was cheaper per run but scored 0.8854 and was twice as slow.
- **Check.** Flash-Lite minimal and low are within 0.02 and cost the same, so I kept minimal. Luna was the cheapest at $0.0007 but scored 0.8136.
- **Write.** Sonnet scored 0.9909 against 0.9744 for Haiku off. That is inside the tie band, and Sonnet cost $0.0200 against $0.0010 for the step, which also puts the whole run over the per-match cost line. Haiku low scored the same as Haiku off but cost more and was slower ($0.0016 and 7.0 s against $0.0010 and 2.5 s). Haiku off.

The combination went end to end first, then three full passes. On the tuning jobs the final run scored 0.830 for recall of the right sections, 0.913 for correctly saying "not covered", and 0.973 for faithfulness, with 63 of 63 code checks passing. Those numbers are in the results on the Evals page. One of the three passes was below the line on "not covered", which I report rather than hide.

## Why a cost line matters

For most of this project I was spending free API credits, and when the credits are free it is easy to keep reaching for the strongest model at every step. The table shows what that would have looked like: Sonnet on reading and writing costs about twenty times what Haiku does at each of those steps, for a gain of nothing on reading and 0.0165 on writing.

The Gemini and OpenAI calls were always paid. The free Anthropic credits stop on 2 December, and after that the Anthropic calls become real money too. The site is not live yet, so this is the bill I am preparing for. So I measured a cost for each option, not only a score, and I made the cost part of the rule instead of a feeling. A live job-match run on the final combination averaged about $0.0091, measured over three passes of 21 jobs.

That number is also the thing I can reason about. If someone asks "what if a lot of people use it?", I do not have to guess. I can multiply by a number I measured, and I can see which step is worth watching: matching is the largest share per run, so any change there has the most effect on the bill.
