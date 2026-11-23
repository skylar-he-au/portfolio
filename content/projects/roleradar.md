---
title: RoleRadar
summary: An AI job-search assistant for Australian tech roles that matches jobs to your resume and tailors resumes without inventing experience.
updated: 2026-11-17
rag: true
kicker: Project · live at roleradar.org
proof:
  - value: 154 / 164
    label: tier checks agreed with labelled answers
  - value: 17 / 17
    label: resume runs kept every fact
  - value: 332
    label: company job boards
---

## In 30 seconds

RoleRadar is an AI job-search assistant for people looking for tech roles in Australia. I built it alone, with most of the code written by Claude Code, and it is live at https://roleradar.org. It collects jobs from public sources, rates how well each one fits your resume, tailors a resume for a job, and answers questions about your own jobs and applications.

- 154 of 164 tier checks in the fit-check eval agreed with labelled answers (run 2026-11-16).
- 17 of 17 resume-tailoring eval runs kept every fact from the original resume (run 2026-11-16).
- 332 company job boards on the daily source list (as of 2026-11-17).

Stack: React 19, Redux Toolkit, Express 5 on AWS Lambda, MongoDB Atlas, OpenAI API. The code is in private repositories, so this page has no code links.

## The problem RoleRadar solves

RoleRadar exists because job hunting in Australian tech means sorting a long, noisy stream of ads, and because AI help with that is easy to get wrong in two ways. I wanted a tool that is honest about both.

The first is sourcing. Job ads are spread across company career pages, job boards and alert emails. RoleRadar pulls them from public company boards and official job APIs, and reads the alert emails a user forwards to it. It does not scrape sites, because sites block bots and scraping breaks their terms.

The second is trust in the AI. A language model that scores "fit" out of 100 gives numbers that look precise and are not, and one that rewrites a resume will happily add skills the candidate never had, which gets people caught in interviews. So the product is built around a few rules the product is built around:

- The scope is Australian IT roles.
- Fit is shown as three tiers (Strong, Fair, Weak) with a reason for each, not as a percentage.
- Every fact the AI reads from an ad or a resume must come with a quote that code can find in the source.
- RoleRadar drafts things for the user. It does not send applications on the user's behalf.

RoleRadar's matching, tailoring and chat are each built around these rules.

## How RoleRadar is built

RoleRadar is a serverless AWS application. The front end is React 19 with Redux Toolkit. The back end is Express 5 on three Lambda functions: one behind an HTTP API for normal requests, one with its own function URL for streaming chat, and a worker for long jobs, with CloudFront in front. Data is in MongoDB Atlas, files in S3, email goes through SES, and EventBridge Scheduler starts the daily jobs. AI calls go to the OpenAI API.

I chose serverless over the always-on server setup I had been taught, because one person running a product with real data and email needs cost to follow usage. Cold starts were the price. A keep-warm ping every five minutes took the first request after an idle period from 4.1 seconds to 0.40 seconds in one live measurement on 2026-11-08, and I have not re-run it since.

Jobs arrive through a daily pipeline that EventBridge starts at 6:30am Sydney time. It reads 332 company boards across five applicant-tracking-system APIs (Greenhouse, Ashby, Workable, SmartRecruiters and Lever), the Adzuna API, and forwarded SEEK and LinkedIn alert emails that SES receives. The 332 is the size of the list as of 2026-11-17, not a count of boards that returned jobs that day.

Merging sources needs de-duplication, and my earlier key was too aggressive: in a research run on 2026-10-12 it dropped 45 jobs from a 385-job test batch taken from the 73 boards I had then. I changed the key to company, title and overlapping location, and let a full ad replace earlier summary-only copies. On the same batch the new key dropped 1 job instead of 45, and that one was a real duplicate. I have not re-run it since.

Long work goes to the worker Lambda by an asynchronous invoke rather than a message queue.

## How RoleRadar matches jobs

RoleRadar matches jobs with two layers: an AI model reads each full ad once, and plain code decides the fit tier. I split it this way after weighing three options: let the model score fit, use rules only, or combine a model that reads with rules that decide.

In the first layer, gpt-6-luna turns each ad into structured fields such as required skills, level and work-rights requirements. Every extracted fact has to include a verbatim quote from the ad, and code drops or downgrades any quote it cannot find there. Each ad is read once and the result is shared by all users, so reading ads costs the same however many users there are.

The second layer runs per user, in code. Hard gates come first: for example a work-rights mismatch, or a senior role when the user did not choose senior, rates a job Weak with the reason stated. Then four scored checks (skills, level, work rights and hours, location) set the tier, with several caps that stop a job rating Strong when the ad leaves out important information. Because tiers come from code, every reason traces back to a line in the ad.

On real data the rules alone over-rated matches. When I walked six of my own live Strong matches (from 44 matches, 2026-11-10) through a resume check, five dropped to Weak and one to Fair. So I added a per-user resume check of the ad's must-have requirements, again with quotes that code verifies.

The result on 2026-11-16, prompt version 12: 154 of 164 tier checks matched the labels, and 0 of 462 quoted pieces of evidence were missing from their source. Caveats: that run also had 1 error and 5 cases whose tier changed between runs. Of the labels, 39 were drafted by Claude for me to change, 13 I confirmed myself, and 5 are demo cases, so it measures agreement with labels that are mostly AI-drafted and partly confirmed by me, not with hiring outcomes. I picked low reasoning effort after comparing it with none and medium; medium scored 152 of 164 at 1.9 times the cost.

## How RoleRadar tailors a resume without inventing experience

RoleRadar's resume tailoring writes a version of the user's resume for a specific job and is built to stop it adding experience the user does not have. The original resume is never changed. The user reviews the new version and saves it as a separate one.

The rule is that the AI may use the ad's wording to describe things the user really did, and may reorder and emphasise, but may not add technologies, years or results that are not in the original.

I treated this as a checking problem with three options: trust one model, have a second model check it, or check with code first. I chose code first, then a model:

1. gpt-5.6-luna rewrites resume lines and must cite the source lines it used.
2. Code rejects anything it can verify is new: numbers, source line references, skill lists, and claims of "production" use that the source never makes.
3. A narrow term-check call must quote the resume to justify a term.
4. gpt-5.6-terra checks everything else.

If lines are still flagged after three attempts, they revert to the user's own wording. I use gpt-5.6-terra for the checks because, in a 2026-10-04 comparison, both models caught 12 of 12 planted fabrications but gpt-5.6-luna raised a false alarm in every run and needed more rewrites (1.26 drafts on average against 1.07).

In the committed eval run of 2026-11-16 (prompt version 2), 17 of 17 runs kept every fact, 34 of 34 pieces of evidence were kept, and no lines had to be reverted. The run cost under ten cents. An earlier test on 2026-10-17, before this prompt version, caught 13 of 13 planted fabrications with no false alarms on 11 clean cases. I have not committed a result that measures fabrication detection on the current prompt, so I do not quote one.

## How RoleRadar tests its AI: evals and prompt versions

RoleRadar has an evaluation harness because prompt and model changes quietly break AI features, and I would not notice by using the app. It runs as GitHub Actions workflows as of 2026-11-17 with 10 suites: job rules, match tiers, job-ad reading, alert emails, reply emails, resume check, resume tailoring, resume goals, fit check and interview. Eight of them call OpenAI.

- On every push, a code-only subset runs. It needs no API calls.
- Once a week (Monday, Sydney time), the suites run against pass lines.
- Every run has its own spend cap, and that cap applies even when daily caps are switched off.
- Chat and brief suites run on demand.

I scored AI output with labelled real data and exact field comparison, not with a model judge called through the API, so scores are cheap and repeatable. Free-text rubric lines are graded in a Claude Code review session instead of through the API. I moved to that after one day of evals with an API grader emptied the OpenAI account balance.

Prompts live in code, one module per feature, and a unit test fails if a prompt changes without a new version number. As of 2026-11-17, twelve prompts are versioned, and every AI-call log line and eval result records the prompt version it ran on. That is how I can say which results still describe the code that is live. For example, the job-ad reading eval last ran on version 8 (1,010 of 1,064 field checks over 49 labelled ads, 2026-11-04) while the current prompt is version 10, so I treat that number as old.

Things I have not measured: production latency, and the full chat set on the current prompt.

## The chat assistant and read-only tools

RoleRadar's chat assistant answers questions about a user's own jobs and applications, such as which saved job to apply to first or what is happening with an application. It streams answers to the browser over server-sent events from a dedicated Lambda with a function URL.

The assistant is a hand-written function-calling loop on gpt-5.6-luna. It has five read-only lookup tools plus cards that propose a change, which the user must confirm. It cannot change data itself. The user's id comes from code, never from the model, so a prompt cannot ask for someone else's data. Text copied from job ads and emails is wrapped as data, not instructions.

I wrote the loop myself because the agent SDKs I looked at sent traces to the vendor by default and had no hook for per-user spend limits. As of 2026-11-17, a turn is limited to six model calls and 75 seconds. Memory is the last 20 messages plus up to 20 notes the user has approved.

On 2026-11-16, 23 questions about the current page, each run three times, passed 877 of 879 code checks and 67 of 69 cases. The 147 rubric lines were all graded as met, at about 0.09 US cents and 3.9 seconds per turn. That run covers page-context questions only. The earlier full run on an older prompt (2026-11-11) still failed some ranking, how-to and status questions, and I have not re-run the full set on the current prompt.

I also built a remote, read-only MCP server with its own OAuth 2.1 authorisation (PKCE, consent screen, per-user limits). Users can ask their own Claude about their RoleRadar jobs through it, with no AI cost to RoleRadar. It has been live since 2026-11-03 and I have not measured it with real users yet.

## How RoleRadar stays safe and affordable

RoleRadar keeps one person's data away from another's, and keeps AI cost bounded, by checks in code rather than by trusting prompts.

Security and data:

- The user id comes only from a signed token that can be revoked. Changing a password makes older tokens stop working.
- Another user's record answers 404, as if it did not exist.
- File uploads go straight to S3 and the server checks that the key sits under that user's own prefix.
- As of 2026-11-17, request bodies are validated by 37 zod schemas.
- Lambda instances share no memory, so in-memory rate limits would do nothing. Rate-limit counters live in MongoDB instead, with API Gateway throttling in front. I chose that over Redis to avoid the cost and a VPC.

AI cost:

- Every model call goes through one in-house gateway that applies per-user and site-wide daily limits, feature switches, timeouts and cost logging.
- Spend is checked before each call, so a limit stops work instead of reporting overspend afterwards.
- I routed models by eval: gpt-6-luna reads ads, gpt-5.6-luna chats and writes, and gpt-5.6-terra only does fabrication checks. gpt-6-luna read ads as well as gpt-5.6-luna (1,010 against 1,000 of 1,064 checks in the 2026-11-04 run) at half the price.
- Prompts put instructions first and per-turn data after, so the prefix caches: 93 percent of chat input was cached in the 2026-11-16 chat eval run.

What is missing: there is no fallback model, and I have not tried the Batch API for the daily ad reading.

## How I built RoleRadar with Claude Code

Most of RoleRadar's code was written by Claude Code, and I did not hand-write it. I set the scope and rules, made the architecture and model choices, and reviewed changes. Claude co-authored 423 of 457 commits in the API repository and 269 of 299 in the web repository (as of 2026-11-17).

The work is about directing and checking an AI developer, and the setup that makes that safe is part of what I built:

- A rules file records who decided each rule (me, me agreeing to the AI's suggestion, or the AI) and requires my approval before any new spending.
- Every product rule goes in a spec file with a line giving the reason, who decided and the date.
- Custom skills split large work into pieces and hand it over between sessions, and hooks tell the AI to hand over when its context is 30 percent used.
- As of 2026-11-17, for six feature plans I had Codex, run in a sandbox without network access, review the plan before any code was written. It raised 96 findings across them. Each was checked and not all were applied.

Tests and evals decide whether the result works. The API has 2,576 tests in 127 suites and the web app has 1,270 in 72 files (both run 2026-11-17). Both run on every push. Deploys are manual on purpose: I start each one by hand in GitHub Actions.

I have no measured number for what the hooks prevented, because they keep no log.
