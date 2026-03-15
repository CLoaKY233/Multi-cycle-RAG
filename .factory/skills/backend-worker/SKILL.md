---
name: backend-worker
description: Handles backend logic, database integrations, and Python CLI enhancements.
---

# backend-worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill
Use this skill for implementing database schema updates, Python logic, asynchronous operations, and Typer/Rich CLI feature enhancements.

## Work Procedure

1. **Investigate Context:** Check the related files (e.g., `src/vectorstore/`, `src/rag/`, `rag.py`) to understand current data flows.
2. **Test-Driven Approach:** 
   - Before implementing new logic (like cache querying or concurrent execution), write standalone python test scripts or standard pytest assertions to verify behavior in isolation.
   - Run the tests to ensure they fail initially (red).
3. **Implementation:** 
   - Apply the changes required.
   - For database schema changes, ensure `.surql` files are properly formatted.
   - For asyncio changes, ensure no race conditions exist.
4. **Verification:**
   - Run the tests and ensure they pass (green).
   - Manually verify by executing `uv run rag.py` with appropriate flags to simulate user interaction.
5. **Handoff:** Prepare a comprehensive JSON handoff documenting exact commands run and output observed.

## Example Handoff

```json
{
  "salientSummary": "Implemented asyncio.gather() in reflexion_engine.py and removed sleep bottlenecks. Added tests for parallel execution which all pass. Manually verified CLI performance.",
  "whatWasImplemented": "Replaced sequential db and web search await calls with asyncio.gather(). Removed await asyncio.sleep(0.1) from the reflexion loop.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "python test_parallelism.py",
        "exitCode": 0,
        "observation": "All async gather assertions passed successfully."
      }
    ],
    "interactiveChecks": [
      {
        "action": "Ran uv run rag.py chat and submitted a query.",
        "observed": "Response time dropped by 40% and web search logs show parallel execution."
      }
    ]
  },
  "tests": {
    "added": [
      {
        "file": "test_parallelism.py",
        "cases": [
          {
            "name": "test_concurrent_fetches",
            "verifies": "Ensures DB and web requests resolve concurrently without blocking."
          }
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator
- If the SurrealDB remote instance is unreachable or credentials in `.env` are invalid.
- If dependency conflicts prevent `uv sync` from completing.
