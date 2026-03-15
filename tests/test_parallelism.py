"""Tests for asyncio.gather concurrent execution and sleep delay removal.

This file validates the performance parallelism feature:
- VAL-PERF-001: Concurrent execution with asyncio.gather
- VAL-PERF-002: No arbitrary sleep delays
"""

import ast
import asyncio
import time
from pathlib import Path
from typing import List, Optional

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_function_node(tree: ast.AST, name: str) -> Optional[ast.AsyncFunctionDef]:
    """Return the first top-level (or class-member) async function named *name*."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    return None


def _enclosing_function_names(tree: ast.AST, target_lineno: int) -> List[str]:
    """Return the names of all function/async-function nodes that contain
    the given line number (used to determine 'allowed' sleep contexts)."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            if start <= target_lineno <= end:
                names.append(node.name)
    return names


def _collect_sleep_calls(tree: ast.AST) -> List[ast.Await]:
    """Return all AST Await nodes that call asyncio.sleep(...)."""
    sleeps = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        # Accept both `asyncio.sleep(...)` (Attribute) forms
        if isinstance(func, ast.Attribute) and func.attr == "sleep":
            sleeps.append(node)
    return sleeps


def _has_gather_in_function(func_node: ast.AST) -> bool:
    """Return True if *func_node* contains an asyncio.gather call."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "gather":
                return True
    return False


# ---------------------------------------------------------------------------
# Sleep-delay tests
# ---------------------------------------------------------------------------


class TestSleepDelaysRemoved:
    """Test that hardcoded sleep delays have been removed from reflexion loop."""

    def test_no_sleep_in_reflexion_engine(self):
        """Verify no arbitrary asyncio.sleep calls in reflexion_engine.py.

        Only sleeps inside explicitly-allowed helper functions are permitted.
        """
        reflexion_path = (
            Path(__file__).parent.parent / "src" / "rag" / "reflexion_engine.py"
        )
        source = reflexion_path.read_text()
        tree = ast.parse(source)

        # Functions allowed to contain asyncio.sleep (e.g. rate-limiting helpers)
        allowed_functions = {"_stream_cached_result", "_rate_limit_web_search"}

        sleep_nodes = _collect_sleep_calls(tree)
        for node in sleep_nodes:
            enclosing = _enclosing_function_names(tree, node.lineno)
            # If the sleep is inside *any* allowed function, skip it
            if any(fn in allowed_functions for fn in enclosing):
                continue
            pytest.fail(
                f"Found asyncio.sleep at line {node.lineno} "
                f"in context {enclosing} — remove or move to an allowed function."
            )

    def test_no_sleep_in_main_loop_body(self):
        """Verify the reflexion main loop (query_with_reflexion_stream) has no sleep."""
        reflexion_path = (
            Path(__file__).parent.parent / "src" / "rag" / "reflexion_engine.py"
        )
        source = reflexion_path.read_text()
        tree = ast.parse(source)

        func_node = _get_function_node(tree, "query_with_reflexion_stream")
        if func_node is None:
            pytest.skip("query_with_reflexion_stream not found in reflexion_engine.py")
            return  # unreachable — narrows type for static analysis

        sleep_nodes = _collect_sleep_calls(func_node)
        if sleep_nodes:
            lines = [n.lineno for n in sleep_nodes]
            pytest.fail(
                f"asyncio.sleep found inside query_with_reflexion_stream "
                f"at lines {lines} — should be removed."
            )


# ---------------------------------------------------------------------------
# Concurrent-execution (gather) tests
# ---------------------------------------------------------------------------


class TestConcurrentExecution:
    """Test that asyncio.gather is used for concurrent DB and web retrieval."""

    def test_similarity_search_combined_uses_gather(self):
        """Verify similarity_search_combined uses asyncio.gather for parallel searches."""
        store_path = (
            Path(__file__).parent.parent / "src" / "vectorstore" / "surrealdb_store.py"
        )
        source = store_path.read_text()
        tree = ast.parse(source)

        func_node = _get_function_node(tree, "similarity_search_combined")
        assert func_node is not None, (
            "similarity_search_combined not found in surrealdb_store.py"
        )
        assert _has_gather_in_function(func_node), (
            "asyncio.gather not found inside similarity_search_combined — "
            "concurrent execution not implemented"
        )

    def test_reflexion_engine_uses_gather_for_retrieval(self):
        """Verify query_with_reflexion_stream uses asyncio.gather for concurrent retrieval."""
        reflexion_path = (
            Path(__file__).parent.parent / "src" / "rag" / "reflexion_engine.py"
        )
        source = reflexion_path.read_text()
        tree = ast.parse(source)

        func_node = _get_function_node(tree, "query_with_reflexion_stream")
        assert func_node is not None, (
            "query_with_reflexion_stream not found in reflexion_engine.py"
        )
        assert _has_gather_in_function(func_node), (
            "asyncio.gather not found inside query_with_reflexion_stream — "
            "concurrent DB/web retrieval not implemented"
        )

    @pytest.mark.asyncio
    async def test_concurrent_execution_timing(self):
        """Verify that concurrent execution is faster than sequential."""

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

        assert concurrent_time < sequential_time * 0.7, (
            f"Concurrent ({concurrent_time:.3f}s) should be faster than "
            f"sequential ({sequential_time:.3f}s)"
        )


# ---------------------------------------------------------------------------
# AST-analysis tests (broader coverage)
# ---------------------------------------------------------------------------


class TestASTAnalysis:
    """AST-based analysis for concurrent execution patterns."""

    def test_gather_in_vector_store(self):
        """similarity_search_combined must call asyncio.gather."""
        store_path = (
            Path(__file__).parent.parent / "src" / "vectorstore" / "surrealdb_store.py"
        )
        tree = ast.parse(store_path.read_text())

        func_node = _get_function_node(tree, "similarity_search_combined")
        assert func_node is not None, (
            "similarity_search_combined not found in surrealdb_store.py"
        )

        gather_lines = []
        for node in ast.walk(func_node):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "gather"
            ):
                gather_lines.append(node.lineno)

        assert gather_lines, (
            "asyncio.gather must be used in similarity_search_combined "
            "for concurrent DB searches"
        )
        print(f"Found asyncio.gather at lines: {gather_lines}")

    def test_gather_in_reflexion_engine(self):
        """query_with_reflexion_stream must call asyncio.gather."""
        reflexion_path = (
            Path(__file__).parent.parent / "src" / "rag" / "reflexion_engine.py"
        )
        tree = ast.parse(reflexion_path.read_text())

        func_node = _get_function_node(tree, "query_with_reflexion_stream")
        assert func_node is not None, (
            "query_with_reflexion_stream not found in reflexion_engine.py"
        )

        gather_lines = []
        for node in ast.walk(func_node):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "gather"
            ):
                gather_lines.append(node.lineno)

        assert gather_lines, (
            "asyncio.gather must be used in query_with_reflexion_stream "
            "for concurrent retrieval"
        )
        print(f"Found asyncio.gather at lines: {gather_lines}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
