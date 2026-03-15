"""
Diagnostic script: verifies that Tavily search is correctly configured and working.

Usage:
    uv run python check_tavily.py
"""

import asyncio
import os
import sys


def check_env_file():
    print("\n[1/4] Checking .env file for TAVILY_API_KEY ...")
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        print("  ERROR: .env file not found.")
        return False

    with open(env_path) as f:
        for line in f:
            if line.strip().startswith("TAVILY_API_KEY="):
                value = line.strip().split("=", 1)[1]
                if value and value != "your_tavily_api_key_here":
                    print(f"  OK   : TAVILY_API_KEY found ({value[:12]}...)")
                    return True
                else:
                    print(
                        "  ERROR: TAVILY_API_KEY is a placeholder — update it in .env"
                    )
                    return False

    print("  ERROR: TAVILY_API_KEY not found in .env")
    return False


def check_settings():
    print("\n[2/4] Loading settings via pydantic-settings ...")
    try:
        from src.config.settings import settings

        key = settings.tavily_api_key
        if key:
            print(f"  OK   : settings.tavily_api_key loaded ({key[:12]}...)")
            return True
        else:
            print("  ERROR: settings.tavily_api_key is empty.")
            return False
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False


def check_import():
    print("\n[3/4] Importing TavilyWebSearch ...")
    try:
        from src.websearch import TavilyWebSearch  # noqa: F401

        print("  OK   : TavilyWebSearch imported successfully.")
        return True
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False


async def check_live_search():
    print("\n[4/4] Running a live Tavily search ...")
    try:
        from src.websearch import TavilyWebSearch

        searcher = TavilyWebSearch()
        available = await searcher.is_available()
        if not available:
            print("  ERROR: TavilyWebSearch.is_available() returned False.")
            return False

        results = await searcher.search_and_extract(
            "Python programming language", num_results=2
        )
        if not results:
            print("  ERROR: No results returned.")
            return False

        for r in results:
            status = r.status.value
            words = r.word_count
            print(f"  [{status}] #{r.rank} {r.url[:70]}")
            print(f"           Title  : {r.title[:70]}")
            print(f"           Words  : {words}")
            print()

        successes = [r for r in results if r.status.value == "success"]
        if successes:
            print(
                f"  OK   : {len(successes)}/{len(results)} results with status=success."
            )
            return True
        else:
            print(
                "  WARN : No results had status=success — check content length settings."
            )
            return False

    except Exception as exc:
        print(f"  ERROR: {exc}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    print("=" * 60)
    print(" Tavily Search Diagnostic")
    print("=" * 60)

    steps = [
        check_env_file(),
        check_settings(),
        check_import(),
        await check_live_search(),
    ]

    passed = sum(steps)
    total = len(steps)

    print("=" * 60)
    if passed == total:
        print(f" ALL CHECKS PASSED ({passed}/{total}) — Tavily is ready.")
    else:
        print(f" {total - passed}/{total} CHECK(S) FAILED — see errors above.")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
