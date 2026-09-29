# Urban Heat & Climate Resilience Research Assistant

A multi-agent research assistant focused on urban heat resilience and climate adaptation. Ask about heat risk, public-health impacts, neighborhood vulnerability, or city adaptation strategies. The workflow plans focused investigations with OpenAI, searches a curated set of trusted sources, reviews evidence quality, pauses for your review, and then writes a structured report with source links.

## What it does

- **Planner:** creates three to five focused sub-tasks for the research question.
- **Researcher:** searches for evidence with Tavily.
- **Critic:** rates relevance and completeness, requesting another research round when evidence is weak.
- **Human review:** pauses before synthesis so you can inspect and edit the gathered findings.
- **Synthesizer:** writes a structured report using the reviewed findings.
- **Persistent checkpoints:** stores graph state in SQLite so paused reviews and completed reports survive server restarts.

By default, Tavily is restricted to these urban climate, public-health, government, and research domains:

`climate.gov`, `noaa.gov`, `epa.gov`, `cdc.gov`, `ipcc.ch`, `who.int`, `unhabitat.org`, `wri.org`, `worldbank.org`, `nature.com`, and `sciencedirect.com`.

Set `TAVILY_INCLUDE_DOMAINS` to a comma-separated list to replace the defaults. Searches are restricted to the configured domains.

## Requirements

- Python 3.11 or newer
- An OpenAI API key
- A Tavily API key

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the safe example file, then add your OpenAI and Tavily keys to `.env`:

```bash
cp .env.example .env
```

The language model uses the OpenAI API with `gpt-4o-mini` by default. OpenAI and Tavily require API keys. `.env` is excluded by `.gitignore` and must not be committed.

## Run the web app

```bash
.venv/bin/python webapp.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). If port 8765 is unavailable, choose another local port with `RESEARCH_PORT`:

```bash
RESEARCH_PORT=8766 .venv/bin/python webapp.py
```

The app binds to `127.0.0.1` by default. A research run requires valid OpenAI and Tavily API keys.
Graph checkpoints are saved by default at `data/research_checkpoints.sqlite`. Set `CHECKPOINT_DB_PATH` in `.env` to choose another location. The browser remembers the latest run ID so a paused review or completed report can be restored after restarting the app in the same browser. Keep the database private because it contains research questions and collected findings; it is excluded by `.gitignore`.

The header service checks confirm the configured OpenAI model is available to your project and make a one-result Tavily search to check API access. If a model/API request fails, the app displays a provider-specific action. Runs interrupted during an agent step can be resumed from their last SQLite checkpoint using **Resume saved research**. Tavily health checks make a small search request and may use one search credit.

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

The OpenAI schema-binding test runs without making a model request. To also test actual inference (which uses the OpenAI API), configure a valid key and run:

```bash
RUN_OPENAI_MODEL_TESTS=1 .venv/bin/python -m pytest -m openai
```

## Review and configuration

Before the report is written, the workflow pauses. In the web app, inspect or edit the findings JSON and choose **Approve & write report** to resume synthesis.

- `OPENAI_API_KEY`: required OpenAI API key.
- `OPENAI_MODEL`: OpenAI model (default: `gpt-4o-mini`).
- `CHECKPOINT_DB_PATH`: SQLite workflow checkpoint file (default: `data/research_checkpoints.sqlite`).
- `TAVILY_INCLUDE_DOMAINS`: optional comma-separated allowlist replacing the built-in urban climate sources.
- `RESEARCH_PORT`: optional local web server port (default: `8765`).
