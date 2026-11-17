---
title: Privacy
summary: What happens to the questions you type, who else sees them, and how long each party keeps them.
updated: 2026-11-17
rag: true
---

## Who sees your question

This Privacy page of Skylar's portfolio explains where a question typed into the site goes. Your question is processed by up to five services: Anthropic (Claude rewrites your question and writes the answer), Google (Gemini checks the answer, takes over the other's job if one of the two fails, and Google Cloud Run also hosts this site's backend), OpenAI (embeddings, voice transcription and read-aloud), Cloudflare (bot check and request routing) and Neon (the database that stores the data described below).

## How long the AI providers keep it

This site (Skylar's portfolio) sends your text to outside services, and each one may keep what it receives for a while to detect abuse; I can't turn that off on a personal account. Here is what each one says, as of 2026-11-17:

- **Anthropic** deletes API inputs and outputs within 30 days. Content its automated systems flag as breaking its usage policy may be kept for up to 2 years. It does not use API data to train its models by default.
- **Google** (paid Gemini tier) keeps prompts and responses for a limited period only to detect abuse; Google does not say how long. On the paid tier your text is not used to improve Google's products.
- **OpenAI** keeps embedding and read-aloud requests for up to 30 days and does not keep voice transcription requests. It does not use API data for training.
- **Cloudflare** checks for bots with Turnstile, which looks at signals such as your IP address and browser; Cloudflare does not say how long it keeps them. I turn off Cloudflare's Worker logs.
- **Neon** hosts the database in Sydney. After this site deletes a record, Neon can still restore it for up to 6 hours, and Neon keeps encrypted backups for 30 days.

## What this site stores and for how long

Privacy on this site: your question is stored for 30 days with these fields only: time, question text, answer language, prompt version, which model provider was used, cost, a hash of your IP address for that day, and any thumbs-down you gave. The record does not contain the answer, any pasted job description, your session token or your IP address. Conversation state, which includes answers and pasted job description text, is deleted after 24 hours, and so is the read-aloud audio cache. A scheduled cleanup runs every day and deletes expired data. It also keeps per-day request counters for your session and for the daily IP hash, plus spending counters.

## IP addresses

This site does not save your IP address in its database or logs. It stores a hash of it, made with a secret key that changes every day and is deleted afterwards. The hash lets the site count your requests within one day, and it can't be linked to a hash from another day.

## Logs and tracing

This site does not keep request logs: request logging on the hosting service is switched off, and the server's access log is off. Tracing tools such as LangSmith are not used. The site's own logs only record usage counts and hashes.

## Voice

If you use the microphone on Skylar's portfolio site, the recording is not stored by this site. The transcribed text is then handled like a typed question.

## Memory between visits

Skylar's portfolio site remembers nothing between visits: the assistant remembers only within one conversation, and the conversation is deleted after 24 hours. It does not recognise you or remember anything from an earlier visit.

## Questions used to test the assistant

Questions used to test the assistant on this site (Skylar's portfolio): a question may be rewritten by me to remove anything that could identify you, then used to test the assistant. Rewritten questions can be kept longer than 30 days. The test set is kept only on my own computer, not in any online repository.

## Deleting your data

To have your data on this site (Skylar's portfolio) deleted, email me at TBD (address to be added) and include the exact text of your question so I can find it.
