# Smart Recommendation System for Bank Payment Failures

An intelligent, autonomous AI recommendation agent designed for bank payment failure resolution and manual queue reduction.

## Overview

When payments fail in financial routing systems, they often land in manual review queues. This system autonomously investigates failed transactions by executing relevant diagnostic tools (checking bank availability, verifying account statuses, inspecting account balances, and analyzing IBAN formatting or name transliterations) and generates precise, actionable recommendations:

- **`RETRY_NOW`**: For temporary network/processing hiccups when the destination bank is online and account is active.
- **`RETRY_LATER`**: When destination banks are degraded or down, computing optimal backoff or scheduled retry times.
- **`RETRY_AFTER_UPDATE`**: When payment details are flawed (e.g. IBAN formatting issues with spaces, name mismatches, frozen/closed accounts), with concrete correction steps.
- **`DONT_RETRY`**: When transaction failures are permanent dead ends.

## Features

- **Tool Calling with OpenAI**: Integrated functions for `get_bank_status`, `verify_account`, and `get_balance`.
- **Reasoning Budget**: Constrained reasoning cycles to prevent infinite loops and ensure sub-second to low-latency decisions.
- **Strict JSON Schema**: Standardized JSON response payload with confidence scoring and plain-English explanation.
- **Name Transliteration & IBAN Correction**: Intelligent handling of variations in beneficiary names and formatting artifacts.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export BANK_API_URL="http://127.0.0.1:8001" # Default mock/internal bank API
```

On Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:BANK_API_URL="http://127.0.0.1:8001"
```

## Usage

```bash
python agent.py
```

## License

MIT License
