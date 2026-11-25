# Voice Ordering Agent

A restaurant voice agent: the customer talks in the browser, the agent answers
menu questions, builds the order, reads it back, and sends confirmed orders to
a kitchen screen.

## Stack
- LiveKit Agents + WebRTC (voice in the browser)
- Mistral Voxtral (speech-to-text and text-to-speech)
- Mistral Small + Meta Muse via LLM proxy (chat)
- Mistral embeddings + local vector store (menu search)

## Status
Built step by step. See `PLAN.md` for the checklist and findings so far.

## Findings so far
- Accent test (10 recordings): Voxtral 13.25% WER vs faster-whisper 34.37%
- Word "kottu" breaks ASR -> fixed by fuzzy matching against the menu
- Evaluation must normalize numbers ("6.30" == "six thirty")

## Getting started
Ask me — this repo is a learning project, setup notes land here after step 14.