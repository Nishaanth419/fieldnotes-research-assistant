# Urban Heat & Climate Resilience Research Assistant

A local, multi-agent research assistant focused on urban heat resilience and climate adaptation. Ask about heat risk, public-health impacts, neighborhood vulnerability, or city adaptation strategies. The workflow plans focused investigations, searches a curated set of trusted sources, reviews evidence quality, pauses for your review, and then writes a structured report with source links.

## What it does

- **Planner:** creates three to five focused sub-tasks for the research question.
- **Researcher:** searches for evidence with Tavily.
- **Critic:** rates relevance and completeness, requesting another research round when evidence is weak.
- **Human review:** pauses before synthesis so you can inspect and edit the gathered findings.
- **Synthesizer:** writes a structured report using the reviewed findings.

By default, Tavily is restricted to these urban climate, public-health, government, and research domains:

`climate.gov`, `noaa.gov`, `epa.gov`, `cdc.gov`, `ipcc.ch`, `who.int`, `unhabitat.org`, `wri.org`, `worldbank.org`, `nature.com`, and `sciencedirect.com`.

Set `TAVILY_INCLUDE_DOMAINS` to a comma-separated list to replace the defaults. Searches are restricted to the configured domains.

## Requirements

- Python 3.11 or newer
- A Tavily API key
- [Ollama](https://ollama.com/) with the `llama3.1:8b` model installed

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
```

Copy the safe example file, then add your Tavily key to `.env`:

```bash
cp .env.example .env
```

The language model runs locally through Ollama. Tavily requires an API key for web searches. `.env` is excluded by `.gitignore` and must not be committed.

## Run the web app

```bash
.venv/bin/python webapp.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). If port 8765 is unavailable, choose another local port with `RESEARCH_PORT`:

```bash
RESEARCH_PORT=8766 .venv/bin/python webapp.py
```

The app binds to `127.0.0.1` by default. Ollama must be running locally with the selected model; a research run also requires a valid Tavily key.

If Ollama is not running, start it in another terminal with `ollama serve`.

## Run the terminal interface

```bash
.venv/bin/python main.py
```

## Run automated tests

Install development dependencies and run the deterministic test suite:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

The Ollama schema-binding test runs without making a model request. To also test actual local inference, ensure Ollama is running and the configured model is installed, then run:

```bash
RUN_LOCAL_MODEL_TESTS=1 .venv/bin/python -m pytest -m ollama
```

## Review and configuration

Before the report is written, the workflow pauses. In the web app, inspect or edit the findings JSON and choose **Approve & write report** to resume synthesis.

- `OLLAMA_MODEL`: local Ollama model (default: `llama3.1:8b`).
- `OLLAMA_BASE_URL`: local Ollama server URL (default: `http://127.0.0.1:11434`).
- `TAVILY_INCLUDE_DOMAINS`: optional comma-separated allowlist replacing the built-in urban climate sources.
- `RESEARCH_PORT`: optional local web server port (default: `8765`).
