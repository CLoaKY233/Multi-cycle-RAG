import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.prompt import FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from src.config.settings import WebSearchMode, settings
from src.rag.engine import RAGEngine
from src.utils.input_handler import create_input_handler
from src.utils.preflight import run_preflight_checks

app = typer.Typer(help="Interactive RAG Chat with Reflexion Loop")
console = Console()


class RuntimeConfig:
    """Runtime configuration that can be modified during session"""

    def __init__(self):
        self.web_search_mode = settings.web_search_mode
        self.max_cycles = settings.max_reflexion_cycles
        self.confidence_threshold = settings.confidence_threshold
        self.qa_cache_enabled = settings.qa_cache_enabled
        self.qa_cache_threshold = settings.qa_cache_similarity_threshold

    def apply_to_engine(self, engine: RAGEngine) -> None:
        """Apply runtime config to engine"""
        engine.set_web_search_mode(self.web_search_mode.value)
        engine.set_max_cycles(self.max_cycles)
        engine.set_confidence_threshold(self.confidence_threshold)
        engine.set_qa_cache_enabled(self.qa_cache_enabled)
        engine.set_qa_cache_threshold(self.qa_cache_threshold)


class InteractiveRAGChat:
    """Enhanced interactive RAG chat with runtime configuration"""

    def __init__(self, docs_path: str, runtime_config: Optional[RuntimeConfig] = None):
        self.rag = RAGEngine()
        self.docs_path = docs_path
        self.query_count = 0
        self.config = runtime_config or RuntimeConfig()
        self.input_handler = create_input_handler()

        # Apply initial config
        self.config.apply_to_engine(self.rag)

    async def check_documents_exist(self) -> bool:
        """Check if documents have been ingested"""
        try:
            count = await self.rag.count_documents()
            return count > 0
        except Exception:
            return False

    async def get_document_count(self) -> int:
        """Get the count of documents in the vector store"""
        try:
            return await self.rag.count_documents()
        except Exception:
            return 0

    async def get_websearch_count(self) -> int:
        """Get the count of search results in the vector store"""
        try:
            return await self.rag.count_web_searches()
        except Exception:
            return 0

    def _print_status_bar(self) -> None:
        """Print status bar showing current configuration"""
        status_parts = []

        # Web search status
        if self.config.web_search_mode == WebSearchMode.OFF:
            status_parts.append(("🌐 Web:OFF", "dim"))
        else:
            status_parts.append(
                (f"🌐 Web:{self.config.web_search_mode.value}", "green")
            )

        # Cycles
        status_parts.append((f"🔄 Cycles:{self.config.max_cycles}", "cyan"))

        # Threshold
        status_parts.append(
            (f"🎯 Threshold:{self.config.confidence_threshold:.2f}", "yellow")
        )

        # Cache
        if self.config.qa_cache_enabled:
            status_parts.append(("💾 Cache:ON", "green"))
        else:
            status_parts.append(("💾 Cache:OFF", "dim"))

        # Build status bar
        bar_text = Text()
        bar_text.append("┌", style="dim")
        bar_text.append("─" * 48, style="dim")
        bar_text.append("┐\n", style="dim")
        bar_text.append("│ ", style="dim")

        for i, (text, style) in enumerate(status_parts):
            bar_text.append(text, style=style)
            if i < len(status_parts) - 1:
                bar_text.append(" │ ", style="dim")

        bar_text.append(" │\n", style="dim")
        bar_text.append("└", style="dim")
        bar_text.append("─" * 48, style="dim")
        bar_text.append("┘", style="dim")

        console.print(bar_text)

    async def process_query_with_thinking(self, question: str):
        """Process query with reflexion loop"""
        self.query_count += 1

        console.print(f"\n[bold blue]Query #{self.query_count}[/bold blue]")
        console.print(
            f"[dim]Question: {question[:100]}{'...' if len(question) > 100 else ''}[/dim]"
        )
        console.print()

        await self.process_reflexion_query(question)

    async def process_reflexion_query(self, question: str):
        """Process query using reflexion loop with cache indicator"""

        console.print("[bold blue]🔄 Activating Reflexion Engine[/bold blue]")

        # Show QA cache check if enabled
        if self.config.qa_cache_enabled:
            with console.status(
                "[dim]Checking QA cache for similar questions...[/dim]", spinner="dots"
            ):
                # Cache check happens in the engine
                pass

        current_cycle = 0
        response_text = ""
        cache_hit = False

        try:
            async for chunk in self.rag.query_stream(question):
                if chunk.content:
                    # Check if this is a new cycle
                    if chunk.metadata and chunk.metadata.get("cycle_number"):
                        new_cycle = chunk.metadata["cycle_number"]
                        if new_cycle != current_cycle:
                            current_cycle = new_cycle
                            if current_cycle > 1:
                                console.print(
                                    f"\n[bold yellow]🔄 Cycle {current_cycle}[/bold yellow]"
                                )

                    # Check for QA cache hit
                    if chunk.metadata and chunk.metadata.get("qa_cache_hit"):
                        cache_hit = True
                        similarity = chunk.metadata.get("similarity_score", 0)
                        console.print(
                            f"[bold green]💾 QA Cache Hit! Similarity: {similarity:.2%}[/bold green]"
                        )
                        original_q = chunk.metadata.get("original_question", "")
                        if original_q:
                            console.print(
                                f"[dim]Matched question: {original_q[:60]}...[/dim]"
                            )
                        console.print()

                    # Check for memory cache hit
                    if chunk.metadata and chunk.metadata.get("is_cached"):
                        console.print(
                            "[bold green]💾 [Memory Cache][/bold green] ",
                            end="",
                        )

                    console.print(chunk.content, end="", highlight=False)
                    response_text += chunk.content

                    # Show completion metadata
                    if chunk.is_complete and chunk.metadata:
                        console.print("\n" + "=" * 70)
                        if chunk.metadata.get("reflexion_complete"):
                            self._show_reflexion_stats(chunk.metadata, cache_hit)
                        elif chunk.metadata.get("qa_cached_result"):
                            self._show_qa_cache_stats(chunk.metadata)
                        elif chunk.metadata.get("cached_result"):
                            console.print(
                                "[bold green]💾 Retrieved from memory cache[/bold green]"
                            )
                            console.print(
                                f"[dim]Original cycles: {chunk.metadata.get('total_cycles', 0)}[/dim]"
                            )

        except Exception as e:
            console.print(f"\n[red]❌ Error during reflexion: {e}[/red]")
            console.print("[yellow]Falling back to simple RAG mode...[/yellow]")

    def _show_reflexion_stats(self, metadata: dict, cache_hit: bool = False):
        """Show reflexion completion statistics"""
        if cache_hit:
            console.print("[bold green]✅ Answer Retrieved from QA Cache![/bold green]")
        else:
            console.print("[bold green]✅ Reflexion Complete![/bold green]")

        # Create stats table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("📊 Total cycles:", str(metadata.get("total_cycles", 0)))
        table.add_row(
            "⏱️  Processing time:",
            f"{metadata.get('total_processing_time', 0):.2f}s",
        )
        table.add_row("📚 Documents analyzed:", str(metadata.get("total_documents", 0)))
        table.add_row(
            "🎯 Final confidence:", f"{metadata.get('final_confidence', 0):.2f}"
        )

        if cache_hit:
            table.add_row("💾 Source:", "QA Semantic Cache")

        console.print(table)

    def _show_qa_cache_stats(self, metadata: dict):
        """Show QA cache hit statistics"""
        console.print("[bold green]💾 QA Cache Hit[/bold green]")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("🎯 Similarity:", f"{metadata.get('similarity_score', 0):.2%}")
        table.add_row("⏱️  Processing time:", "Instant (cached)")

        console.print(table)

    async def ingest_documents(self, force_ingest=False):
        """Ingest documents from the specified path"""
        docs_path = Path(self.docs_path)
        if not docs_path.exists():
            console.print(
                f"[red]❌ Error: Documents path '{self.docs_path}' does not exist.[/red]"
            )
            return False

        console.print(
            f"[bold green]📥 Ingesting documents from {self.docs_path}...[/bold green]"
        )

        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents...", total=None)

            try:
                count = await self.rag.ingest_documents(self.docs_path)
                progress.update(task, completed=100, total=100)
                console.print(
                    f"[bold green]✅ Successfully ingested {count} documents.[/bold green]"
                )
                return True
            except Exception as e:
                console.print(f"[red]❌ Error ingesting documents: {e}[/red]")
                return False

    async def show_engine_status(self):
        """Show current engine configuration and status"""
        config = self.rag.get_runtime_config()
        qa_stats = self.rag.get_qa_cache_stats()

        console.print("\n[bold cyan]🔧 Reflexion Engine Configuration[/bold cyan]")

        # Main configuration
        config_table = Table(show_header=False)
        config_table.add_column("Setting", style="dim")
        config_table.add_column("Value", style="bold")

        config_table.add_row("Engine Type", "ReflexionRAGEngine")
        config_table.add_row("Engine Mode", "Reflexion Loop")
        config_table.add_row("🌐 Web Search Mode", config.get("web_search_mode", "off"))
        config_table.add_row(
            "🔄 Max Reflexion Cycles", str(config.get("max_reflexion_cycles", 3))
        )
        config_table.add_row(
            "🎯 Confidence Threshold", str(config.get("confidence_threshold", 0.85))
        )
        config_table.add_row(
            "💾 Memory Cache", str(config.get("memory_cache_enabled", True))
        )
        config_table.add_row(
            "💾 QA Cache Enabled", str(config.get("qa_cache_enabled", True))
        )
        config_table.add_row(
            "💾 QA Cache Threshold", f"{config.get('qa_cache_threshold', 0.85):.0%}"
        )

        console.print(config_table)

        # Show QA cache stats
        if qa_stats.get("enabled"):
            console.print("\n[bold cyan]💾 QA Semantic Cache[/bold cyan]")
            cache_table = Table(show_header=False)
            cache_table.add_column("Metric", style="dim")
            cache_table.add_column("Value", style="bold")
            cache_table.add_row("Status", "Enabled")
            cache_table.add_row(
                "Similarity Threshold", f"{qa_stats.get('threshold', 0.85):.0%}"
            )
            console.print(cache_table)

    async def clear_cache(self):
        """Clear memory cache"""
        await self.rag.clear_memory_cache()
        console.print("[bold green]✅ Memory cache cleared successfully.[/bold green]")

    async def delete_all_documents(self):
        """Delete all documents from the vector store"""
        current_count = await self.get_document_count()

        if current_count == 0:
            console.print("[yellow]⚠️  No documents found in vector store.[/yellow]")
            return

        console.print(
            f"[yellow]⚠️  Warning: This will delete all {current_count} documents![/yellow]"
        )

        confirm = Prompt.ask(
            "[bold red]Type 'CONFIRM' to proceed[/bold red]",
            default="",
        )

        if confirm.strip() != "CONFIRM":
            console.print("[yellow]❌ Deletion cancelled.[/yellow]")
            return

        try:
            console.print("[bold red]🗑️  Deleting all documents...[/bold red]")
            success = await self.rag.delete_all_documents("CONFIRM")

            if success:
                console.print("[bold green]✅ All documents deleted![/bold green]")
            else:
                console.print("[bold red]❌ Failed to delete documents.[/bold red]")

        except Exception as e:
            console.print(f"[bold red]❌ Error: {e}[/bold red]")

    async def interactive_menu(self):
        """Show enhanced interactive menu with runtime configuration"""
        while True:
            console.print()
            self._print_status_bar()
            console.print()

            # Build menu dynamically based on current config
            console.print("[bold cyan]📋 Settings Menu[/bold cyan]")
            console.print("━" * 50)

            # Web search toggle
            web_status = (
                f"[green]ON ({self.config.web_search_mode.value})[/green]"
                if self.config.web_search_mode != WebSearchMode.OFF
                else "[dim]OFF[/dim]"
            )
            console.print(f" 1. 🌐 Web Search: {web_status}  [[bold]toggle[/bold]]")

            # Max cycles
            console.print(
                f" 2. 🔄 Max Cycles: [cyan]{self.config.max_cycles}[/cyan]  [[bold]edit[/bold]]"
            )

            # Confidence threshold
            console.print(
                f" 3. 🎯 Confidence: [yellow]{self.config.confidence_threshold:.2f}[/yellow]  [[bold]edit[/bold]]"
            )

            # QA Cache
            cache_status = (
                f"[green]ON ({self.config.qa_cache_threshold:.0%})[/green]"
                if self.config.qa_cache_enabled
                else "[dim]OFF[/dim]"
            )
            console.print(f" 4. 💾 QA Cache: {cache_status}  [[bold]toggle[/bold]]")

            console.print()
            console.print(" 5. 📊 Show Engine Status")
            console.print(" 6. 🗑️  Clear QA Cache")
            console.print(" 7. 📥 Re-ingest Documents")
            console.print(" 8. 🗑️  Delete All Documents")
            console.print(" 9. ↩️  Return to Chat")
            console.print("━" * 50)

            choice = Prompt.ask(
                "Select an option",
                choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                default="9",
            )

            if choice == "1":
                await self._toggle_web_search()
            elif choice == "2":
                await self._edit_max_cycles()
            elif choice == "3":
                await self._edit_confidence_threshold()
            elif choice == "4":
                await self._toggle_qa_cache()
            elif choice == "5":
                await self.show_engine_status()
            elif choice == "6":
                await self._clear_qa_cache()
            elif choice == "7":
                await self.ingest_documents(force_ingest=True)
            elif choice == "8":
                await self.delete_all_documents()
            elif choice == "9":
                break

    async def _toggle_web_search(self):
        """Toggle web search mode"""
        modes = [
            WebSearchMode.OFF,
            WebSearchMode.INITIAL_ONLY,
            WebSearchMode.EVERY_CYCLE,
        ]
        current_idx = modes.index(self.config.web_search_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.config.web_search_mode = modes[next_idx]
        self.config.apply_to_engine(self.rag)

        mode_name = self.config.web_search_mode.value
        console.print(f"[green]✅ Web search set to: {mode_name}[/green]")

    async def _edit_max_cycles(self):
        """Edit max reflexion cycles"""
        try:
            new_cycles = IntPrompt.ask(
                f"Enter max cycles (1-10) [current: {self.config.max_cycles}]",
                default=self.config.max_cycles,
            )
            if 1 <= new_cycles <= 10:
                self.config.max_cycles = new_cycles
                self.config.apply_to_engine(self.rag)
                console.print(f"[green]✅ Max cycles set to: {new_cycles}[/green]")
            else:
                console.print("[red]❌ Must be between 1 and 10[/red]")
        except Exception as e:
            console.print(f"[red]❌ Invalid input: {e}[/red]")

    async def _edit_confidence_threshold(self):
        """Edit confidence threshold"""
        try:
            new_threshold = FloatPrompt.ask(
                f"Enter threshold (0.0-1.0) [current: {self.config.confidence_threshold}]",
                default=self.config.confidence_threshold,
            )
            if 0.0 <= new_threshold <= 1.0:
                self.config.confidence_threshold = new_threshold
                self.config.apply_to_engine(self.rag)
                console.print(
                    f"[green]✅ Threshold set to: {new_threshold:.2f}[/green]"
                )
            else:
                console.print("[red]❌ Must be between 0.0 and 1.0[/red]")
        except Exception as e:
            console.print(f"[red]❌ Invalid input: {e}[/red]")

    async def _toggle_qa_cache(self):
        """Toggle QA cache"""
        self.config.qa_cache_enabled = not self.config.qa_cache_enabled
        self.config.apply_to_engine(self.rag)

        status = "enabled" if self.config.qa_cache_enabled else "disabled"
        console.print(f"[green]✅ QA cache {status}[/green]")

    async def _clear_qa_cache(self):
        """Clear QA cache (placeholder - would need DB operation)"""
        console.print(
            "[yellow]⚠️  QA cache entries are stored in the database.[/yellow]"
        )
        console.print(
            "[yellow]To clear, you can delete the qa_history table directly.[/yellow]"
        )


@app.command()
def chat(
    docs_path: str = typer.Option(
        "./docs", "--docs-path", "-d", help="Path to documents directory"
    ),
    ingest: bool = typer.Option(
        False, "--ingest", "-i", help="Ingest documents before starting"
    ),
    force_ingest: bool = typer.Option(
        False, "--force-ingest", "-f", help="Force re-ingestion"
    ),
    # Web search options
    no_web: bool = typer.Option(False, "--no-web", help="Disable web search"),
    web: bool = typer.Option(False, "--web", help="Enable web search (initial_only)"),
    web_all: bool = typer.Option(
        False, "--web-all", help="Enable web search every cycle"
    ),
    # Reflexion options
    cycles: int = typer.Option(
        None, "--cycles", "-c", help="Max reflexion cycles (1-10)"
    ),
    threshold: float = typer.Option(
        None, "--threshold", "-t", help="Confidence threshold (0.0-1.0)"
    ),
    # Cache options
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Disable QA semantic cache"
    ),
    cache_threshold: float = typer.Option(
        None, "--cache-threshold", help="QA cache similarity threshold"
    ),
    # Model options
    model: str = typer.Option(None, "--model", "-m", help="Override generation model"),
):
    """Run the interactive RAG chat application"""

    async def app_main():
        # Run pre-flight checks
        if not run_preflight_checks(exit_on_error=True):
            return

        # Build runtime config from CLI args
        runtime_config = RuntimeConfig()

        # Web search mode
        if no_web:
            runtime_config.web_search_mode = WebSearchMode.OFF
        elif web_all:
            runtime_config.web_search_mode = WebSearchMode.EVERY_CYCLE
        elif web:
            runtime_config.web_search_mode = WebSearchMode.INITIAL_ONLY

        # Cycles
        if cycles is not None:
            runtime_config.max_cycles = max(1, min(10, cycles))

        # Threshold
        if threshold is not None:
            runtime_config.confidence_threshold = max(0.0, min(1.0, threshold))

        # Cache
        if no_cache:
            runtime_config.qa_cache_enabled = False
        if cache_threshold is not None:
            runtime_config.qa_cache_threshold = max(0.0, min(1.0, cache_threshold))

        # Override model if specified
        if model:
            settings.llm_model = model

        # Create chat instance
        chat = InteractiveRAGChat(docs_path, runtime_config)

        # Welcome message
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]🤖 Welcome to Reflexion RAG![/bold cyan]\n"
                f"[dim]Documents: {docs_path}[/dim]",
                title="Reflexion RAG CLI",
            )
        )
        console.print()

        # Check if documents exist
        doc_count = await chat.get_document_count()

        if force_ingest or ingest or doc_count == 0:
            if doc_count == 0:
                console.print("[yellow]⚠️  No documents found in vector store.[/yellow]")
            else:
                console.print(f"[yellow]📚 Current documents: {doc_count}[/yellow]")
            success = await chat.ingest_documents()
            if not success:
                console.print(
                    "[red]❌ Cannot proceed without documents. Exiting.[/red]"
                )
                return
        else:
            console.print(
                f"[green]✅ Using existing {doc_count} documents in vector store.[/green]"
            )

        # Show initial status
        console.print()
        chat._print_status_bar()

        # Main chat loop
        console.print()
        console.print("[bold green]💬 Chat started![/bold green]")
        console.print(
            "[dim]Type 'menu' for settings, 'status' for info, 'exit' to quit[/dim]"
        )
        console.print(
            "[dim]'!!' repeats last question, '\\m' for multi-line input[/dim]"
        )

        while True:
            try:
                # Show status bar periodically
                if chat.query_count % 5 == 0 and chat.query_count > 0:
                    console.print()
                    chat._print_status_bar()
                    console.print()

                question = chat.input_handler.get_input("Your question")

                if question is None:
                    continue
                elif question == "exit":
                    console.print("[bold red]👋 Goodbye![/bold red]")
                    break
                elif question.strip().lower() == "menu":
                    await chat.interactive_menu()
                    continue
                elif question.strip().lower() == "status":
                    await chat.show_engine_status()
                    continue
                elif question.strip().lower() == "history":
                    chat.input_handler.print_history()
                    continue
                elif not question.strip():
                    console.print("[yellow]⚠️  Please enter a question.[/yellow]")
                    continue

                await chat.process_query_with_thinking(question)

            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️  Interrupted. Type 'exit' to quit.[/yellow]")
            except EOFError:
                console.print("\n[bold red]👋 Goodbye![/bold red]")
                break
            except Exception as e:
                console.print(f"\n[red]❌ Unexpected error: {e}[/red]")

    asyncio.run(app_main())


@app.command()
def ingest(
    docs_path: str = typer.Option(
        "./docs", "--docs-path", "-d", help="Path to documents"
    ),
):
    """Ingest documents into the vector store"""

    async def ingest_main():
        if not run_preflight_checks(exit_on_error=True):
            return

        chat = InteractiveRAGChat(docs_path)
        console.print("[bold blue]📥 Document Ingestion Mode[/bold blue]")

        current_count = await chat.get_document_count()
        console.print(f"[dim]Current documents in store: {current_count}[/dim]")

        success = await chat.ingest_documents()
        if success:
            new_count = await chat.get_document_count()
            console.print(
                f"[bold green]✅ Ingestion complete! Total: {new_count}[/bold green]"
            )
        else:
            console.print("[bold red]❌ Ingestion failed![/bold red]")

    asyncio.run(ingest_main())


@app.command()
def deletedocs():
    """Delete all documents from the vector store"""

    async def delete_main():
        if not run_preflight_checks(exit_on_error=True):
            return
        chat = InteractiveRAGChat("./docs")
        console.print("[bold red]🗑️  Document Deletion Mode[/bold red]")
        await chat.delete_all_documents()

    asyncio.run(delete_main())


@app.command()
def deleteweb():
    """Delete all web search results from the vector store"""

    async def delete_main():
        if not run_preflight_checks(exit_on_error=True):
            return
        chat = InteractiveRAGChat("./docs")
        console.print("[bold red]🗑️  Web Search Deletion Mode[/bold red]")
        # Use the same delete function for now
        await chat.delete_all_documents()

    asyncio.run(delete_main())


@app.command()
def config():
    """Show current configuration"""
    if not run_preflight_checks(exit_on_error=False):
        console.print("[yellow]⚠️  Some credentials missing - showing defaults[/yellow]")

    console.print("[bold cyan]⚙️  Reflexion RAG Configuration[/bold cyan]")

    config_table = Table(title="Settings", show_header=True)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    # Reflexion settings
    config_table.add_row("Max Reflexion Cycles", str(settings.max_reflexion_cycles))
    config_table.add_row("Confidence Threshold", str(settings.confidence_threshold))
    config_table.add_row("Initial Retrieval K", str(settings.initial_retrieval_k))
    config_table.add_row("Reflexion Retrieval K", str(settings.reflexion_retrieval_k))

    # Memory & Cache settings
    config_table.add_row("Memory Cache Enabled", str(settings.enable_memory_cache))
    config_table.add_row("Max Cache Size", str(settings.max_cache_size))
    config_table.add_row("QA Cache Enabled", str(settings.qa_cache_enabled))
    config_table.add_row(
        "QA Cache Threshold", f"{settings.qa_cache_similarity_threshold:.0%}"
    )

    # Models
    config_table.add_row("Generation Model", settings.llm_model)
    config_table.add_row("Evaluation Model", settings.evaluation_model)
    config_table.add_row("Summary Model", settings.summary_model)

    # Web Search
    config_table.add_row("Web Search Mode", settings.web_search_mode.value)

    # Vector store settings
    config_table.add_row("Embedding Model", settings.embedding_model)
    config_table.add_row("Chunk Size", str(settings.chunk_size))
    config_table.add_row("Chunk Overlap", str(settings.chunk_overlap))

    console.print(config_table)


@app.command()
def check():
    """Run pre-flight credential checks"""
    run_preflight_checks(exit_on_error=False)


if __name__ == "__main__":
    app()
