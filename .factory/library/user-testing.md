# User Testing Strategy

## Validation Surface
The primary surface for testing is the CLI command `rag.py`. 
Testing tools should rely on standard terminal executions (`Execute` or `tuistory` equivalent scripts).

## Validation Concurrency
Based on the dry run, each `uv run rag.py` instance consumes ~155MB of RAM.
The development machine has significant headroom. We can safely run up to 5 concurrent instances of the CLI application.

Max concurrent validators: 5
