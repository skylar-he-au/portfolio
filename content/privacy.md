---
title: Privacy
summary: What happens to the questions you type, who else sees them, and how long each party keeps them.
updated: 2026-11-18
rag: true
---

## Who sees your question

This Privacy page of Skylar's portfolio explains where a question typed into the site goes. Your question is processed by up to five services: Anthropic (Claude rewrites your question and writes the answer), Google (Gemini checks the answer, takes over the other's job if one of the two fails, and Google Cloud Run also hosts this site's backend), OpenAI (embeddings, voice transcription and read-aloud), Cloudflare (bot check and request routing) and Neon (the database that stores the data described below).

## How long the AI providers keep it

This site (Skylar's portfolio) sends your text to outside services, and each one may keep what it receives for a while to detect abuse; I can't turn that off on a personal account. Here is what each one says, as of 2026-11-17:

- **Anthropic** deletes API inputs and outputs within 30 days. Content its automated systems flag as breaking its usage policy may be kept for up to 2 years. It does not use API data to train its models by default.
- **Google** (paid Gemini tier) keeps prompts and responses for a limited period only to detect abuse; Google does not say how long. On the paid tier your text is not used to improve Google's products.
- **OpenAI** keeps embedding and read-aloud requests for up to 30 days and does not keep voice transcription requests. It does not use API data for training.
- **Cloudflare** checks for bots with Turnstile, which runs in invisible mode and only when you click into the question box to start asking, not when a page loads. It looks at signals such as your IP address and browser; Cloudflare does not say how long it keeps them. Cloudflare's own description is in its [Turnstile Privacy Addendum](https://www.cloudflare.com/turnstile-privacy-policy/). I turn off Cloudflare's Worker logs.
- **Neon** hosts the database in Sydney. After this site deletes a record, Neon can still restore it for up to 6 hours, and Neon keeps encrypted backups for 30 days.

## What this site stores and for how long

Privacy on this site (Skylar's portfolio): this is everything the site stores about your visit. A scheduled cleanup runs once a day, and the site also deletes expired data whenever it saves a new question. Because the cleanup runs only once a day, each item is deleted at the first cleanup after it expires, which can be up to about a day later.

- **Question record**: your question is stored for 30 days with these fields only: time, question text, answer language, prompt version, which model providers were used, cost, a hash of your IP address for that day, and any thumbs-down you gave. The record does not contain the answer, any pasted job description, your session token or your IP address. It is deleted at the first cleanup after 30 days, so at most about 31 days after you asked.
- **Conversation state**: answers and any pasted job description text are kept in the conversation for 24 hours, and so is the read-aloud audio cache. They are deleted at the first cleanup after 24 hours, so at most about 48 hours after you asked.
- **Session token**: when the bot check passes, the site gives your browser a session token that works for 24 hours. It is kept only in this browser tab (session storage), so closing the tab forgets it, and it is not saved in the site's database. The site counts requests using a hash of the token's random ID, not the token itself.
- **Daily request counters**: how many requests each session (by token hash) and each daily IP hash made that day, used to enforce daily limits. They are deleted after two days.
- **Spending records**: how much money was spent on each AI provider each day, used for monthly totals. They contain no personal data and are kept.
- **Thumbs-down**: if you click the thumbs-down button under an answer, the site sends only that answer's ID and marks the matching question record. The mark lives in that record and is deleted with it after 30 days.

## IP addresses

This site does not save your IP address in its database or logs. The site's server receives it from Cloudflare and turns it into a hash using a secret key that changes every day and is deleted afterwards. The hash lets the site count your requests within one day, and it can't be linked to a hash from another day. The server also passes your IP address to Cloudflare for the Turnstile bot check described above.

## Logs and tracing

This site does not keep request logs: request logging on the hosting service is switched off, and the server's access log is off. Tracing tools such as LangSmith are not used. The site's own logs only record usage counts and hashes.

## Voice

If you use the microphone on Skylar's portfolio site, the recording is not stored by this site. The transcribed text is then handled like a typed question.

## Memory between visits

Skylar's portfolio site remembers nothing between visits: the assistant remembers only within one conversation. A conversation is kept for 24 hours and deleted at the first daily cleanup after that, so within about 48 hours. It does not recognise you or remember anything from an earlier visit.

## Questions used to test the assistant

Questions used to test the assistant on this site (Skylar's portfolio): a question may be rewritten by me to remove anything that could identify you, then used to test the assistant; questions that received a thumbs-down are the first candidates. Rewritten questions can be kept longer than 30 days. The test set is kept only on my own computer, not in any online repository.

## Deleting your data

To have your data on this site (Skylar's portfolio) deleted, email me at TBD (address to be added) and include the exact text of your question so I can find it.
