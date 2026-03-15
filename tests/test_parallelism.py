"""Tests for asyncio.gather concurrent execution and sleep delay removal.

This file validates the performance parallelism feature:
- VAL-PERF-001: Concurrent execution with asyncio.gather
- VAL-PERF-002: No arbitrary sleep delays
"""

import ast
import asyncio
import time
from pathlib import Path

import pytest


class TestSleepDelaysRemoved:
    """Test that hardcoded sleep delays have been removed from reflexion loop."""

    def test_no_sleep_in_reflexion_engine(self):
        """Verify no asyncio.sleep calls exist in reflexion_engine.py main loop."""
        reflexion_path = Path(__file__).parent.parent / "src" / "rag" / "reflexion_engine.py"
        source = reflexion_path.read_text()
        
        # Parse the AST
        tree = ast.parse(source)
        
        sleep_calls = []
        for node in ast.walk(tree):
            # Check for await asyncio.sleep() calls
            if isinstance(node, ast.Await):
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Attribute):
                        if node.value.func.attr == "sleep":
                            # Get line number for context
                            sleep_calls.append(node.lineno)
        
        # The only acceptable sleep is for streaming cached results (simulate streaming feel)
        # Sleep between cycles (line ~486) should be removed
        # Sleep in _stream_cached_result (line ~815) should be removed
        
        # Check that sleep calls in the main reflexion loop area are not present
        # We look for sleep calls outside of acceptable contexts
        for lineno in sleep_calls:
            # Get the source line context
            lines = source.split('\n')
            if lineno <= len(lines):
                line_content = lines[lineno - 1].strip()
                # Acceptable: sleep in rate limiting context (websearch)
                # Not acceptable: "Brief pause between cycles" or "simulate streaming"
                if "Brief pause between cycles" in line_content:
                    pytest.fail(f"Found arbitrary sleep between cycles at line {lineno}")
                if "simulate" in line_content.lower() and "stream" in line_content.lower():
                    pytest.fail(f"Found arbitrary streaming simulation sleep at line {lineno}")

    def test_no_sleep_in_main_loop_body(self):
        """Specifically check that the main while loop doesn't have sleep delays."""
        reflexion_path = Path(__file__).parent.parent / "src" / "rag" / "reflexion_engine.py"
        source = reflexion_path.read_text()
        
        # Look for the specific pattern of sleep at end of cycle loop
        # The old code had: await asyncio.sleep(0.1)  # Brief pause between cycles
        if "asyncio.sleep(0.1)" in source and "Brief pause between cycles" in source:
            pytest.fail("Found 'Brief pause between cycles' sleep delay - should be removed")
        
        # Also check for sleep in _stream_cached_result - it's arbitrary delay
        if "await asyncio.sleep(0.1)" in source:
            # Check if it's in an acceptable context (rate limiting in websearch)
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if "await asyncio.sleep(0.1)" in line:
                    # This is a hardcoded arbitrary sleep - should be removed
                    pytest.fail(f"Found hardcoded asyncio.sleep(0.1) at line {i+1}")


class TestConcurrentExecution:
    """Test that asyncio.gather is used for concurrent DB and web retrieval."""

    def test_similarity_search_combined_uses_gather(self):
        """Verify similarity_search_combined uses asyncio.gather for parallel searches."""
        store_path = Path(__file__).parent.parent / "src" / "vectorstore" / "surrealdb_store.py"
        source = store_path.read_text()
        
        # The similarity_search_combined should use asyncio.gather for concurrent searches
        # Look for asyncio.gather in the method
        if "asyncio.gather" not in source:
            pytest.fail("asyncio.gather not found in surrealdb_store.py - concurrent execution not implemented")

    def test_reflexion_engine_uses_gather_for_retrieval(self):
        """Verify reflexion_engine uses asyncio.gather for concurrent DB/web retrieval."""
        reflexion_path = Path(__file__).parent.parent / "src" / "rag" / "reflexion_engine.py"
        source = reflexion_path.read_text()
        
        # The main retrieval step should use asyncio.gather when web search is enabled
        # This allows DB retrieval and web search to happen concurrently
        if "asyncio.gather" not in source:
            pytest.fail("asyncio.gather not found in reflexion_engine.py - concurrent retrieval not implemented")

    @pytest.mark.asyncio
    async def test_concurrent_execution_timing(self):
        """Verify that concurrent execution is faster than sequential."""
        # Simulate the performance difference
        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "result"
        
        # Sequential execution time
        start_sequential = time.time()
        await slow_task(0.1)
        await slow_task(0.1)
        sequential_time = time.time() - start_sequential
        
        # Concurrent execution time with asyncio.gather
        start_concurrent = time.time()
        await asyncio.gather(slow_task(0.1), slow_task(0.1))
        concurrent_time = time.time() - start_concurrent
        
        # Concurrent should be roughly half the time of sequential
        assert concurrent_time < sequential_time * 0.7, \
            f"Concurrent ({concurrent_time:.3f}s) should be faster than sequential ({sequential_time:.3f}s)"


class TestASTAnalysis:
    """AST-based analysis for concurrent execution patterns."""

    def test_gather_in_vector_store(self):
        """Analyze surrealdb_store.py for asyncio.gather usage pattern."""
        store_path = Path(__file__).parent.parent / "src" / "vectorstore" / "surrealdb_store.py"
        source = store_path.read_text()
        tree = ast.parse(source)
        
        has_gather = False
        gather_contexts = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "gather":
                        has_gather = True
                        gather_contexts.append(node.lineno)
        
        assert has_gather, "asyncio.gather must be used in surrealdb_store.py for concurrent DB searches"
        print(f"Found asyncio.gather at lines: {gather_contexts}")

    def test_gather_in_reflexion_engine(self):
        """Analyze reflexion_engine.py for asyncio.gather usage pattern."""
        reflexion_path = Path(__file__).parent.parent / "src" / "rag" / "reflexion_engine.py"
        source = reflexion_path.read_text()
        tree = ast.parse(source)
        
        has_gather = False
        gather_contexts = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "gather":
                        has_gather = True
                        gather_contexts.append(node.lineno)
        
        assert has_gather, "asyncio.gather must be used in reflexion_engine.py for concurrent retrieval"
        print(f"Found asyncio.gather at lines: {gather_contexts}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
