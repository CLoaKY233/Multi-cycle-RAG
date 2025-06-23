import asyncio
from pathlib import Path

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
from rich.prompt import Prompt
from rich.table import Table

from src.config.settings import settings
from src.rag.engine import RAGEngine

app = typer.Typer(help="Interactive RAG Chat with Reflexion Loop")
console = Console()


class InteractiveRAGChat:
    def __init__(self, docs_path: str):
        self.rag = RAGEngine()
        self.docs_path = docs_path
        self.query_count = 0

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

    async def process_query_with_thinking(self, question: str):
        """Process query with reflexion loop"""
        self.query_count += 1

        console.print(f"\n[bold blue]Query #{self.query_count}[/bold blue]")
        console.print(f"[dim]Question: {question}[/dim]")
        console.print()

        await self.process_reflexion_query(question)

    async def process_reflexion_query(self, question: str):
        """Process query using reflexion loop"""

        console.print("[bold blue]🔄 Activating Reflexion Engine[/bold blue]")

        # Get engine info
        engine_info = self.rag.get_engine_info()
        console.print(
            f"[dim]Max cycles: {engine_info.get('max_reflexion_cycles', 5)}[/dim]"
        )
        console.print(
            f"[dim]Confidence threshold: {engine_info.get('confidence_threshold', 0.8)}[/dim]"
        )
        console.print(
            f"[dim]Memory cache: {engine_info.get('memory_cache_enabled', False)}[/dim]"
        )
        console.print()

        # Show real-time reflexion process
        console.print("=" * 70)
        console.print("[bold cyan]🤖 AI Reflexion Process[/bold cyan]")
        console.print("=" * 70)

        current_cycle = 0
        response_text = ""

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

                    # Check for cached results
                    if chunk.metadata and chunk.metadata.get("is_cached"):
                        console.print(
                            "[bold green]💾 [Cached Result][/bold green] ",
                            end="",
                        )

                    console.print(chunk.content, end="", highlight=False)
                    response_text += chunk.content

                    # Show completion metadata
                    if chunk.is_complete and chunk.metadata:
                        console.print("\n" + "=" * 70)
                        if chunk.metadata.get("reflexion_complete"):
                            self._show_reflexion_stats(chunk.metadata)
                        elif chunk.metadata.get("cached_result"):
                            console.print(
                                "[bold green]💾 Retrieved from cache[/bold green]"
                            )
                            console.print(
                                f"[dim]Original cycles: {chunk.metadata.get('total_cycles', 0)}[/dim]"
                            )
                            console.print(
                                f"[dim]Original processing time: {chunk.metadata.get('total_processing_time', 0):.2f}s[/dim]"
                            )

        except Exception as e:
            console.print(f"\n[red]❌ Error during reflexion: {e}[/red]")
            console.print("[yellow]Falling back to simple RAG mode...[/yellow]")

    def _show_reflexion_stats(self, metadata: dict):
        """Show reflexion completion statistics"""
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
        table.add_row("💾 Memory cached:", str(metadata.get("memory_cached", False)))

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
        engine_info = self.rag.get_engine_info()

        console.print("\n[bold cyan]🔧 Reflexion Engine Configuration[/bold cyan]")

        # Main configuration
        config_table = Table(show_header=False)
        config_table.add_column("Setting", style="dim")
        config_table.add_column("Value", style="bold")

        config_table.add_row("Engine Type", engine_info.get("engine_type", "Unknown"))
        config_table.add_row("Engine Mode", "Reflexion Loop")
        config_table.add_row(
            "Max Reflexion Cycles",
            str(engine_info.get("max_reflexion_cycles", 5)),
        )
        config_table.add_row(
            "Confidence Threshold",
            str(engine_info.get("confidence_threshold", 0.8)),
        )
        config_table.add_row(
            "Memory Cache",
            str(engine_info.get("memory_cache_enabled", False)),
        )

        console.print(config_table)

        # Show memory stats if available
        if engine_info.get("memory_stats"):
            memory_stats = engine_info["memory_stats"]
            if not memory_stats.get("cache_disabled"):
                console.print("\n[bold cyan]💾 Memory Cache Statistics[/bold cyan]")
                memory_table = Table(show_header=False)
                memory_table.add_column("Metric", style="dim")
                memory_table.add_column("Value", style="bold")

                memory_table.add_row(
                    "Cache Size",
                    f"{memory_stats.get('size', 0)}/{memory_stats.get('max_size', 0)}",
                )
                memory_table.add_row(
                    "Hit Rate", f"{memory_stats.get('hit_rate', 0):.2%}"
                )
                if memory_stats.get("oldest_entry"):
                    memory_table.add_row(
                        "Oldest Entry",
                        f"{memory_stats.get('oldest_entry', 0):.1f}s ago",
                    )

                console.print(memory_table)

    async def clear_cache(self):
        """Clear memory cache if available"""
        await self.rag.clear_memory_cache()
        console.print("[bold green]✅ Memory cache cleared successfully.[/bold green]")

    async def delete_all_documents(self):
        """Delete all documents from the vector store"""
        # Show current document count
        current_count = await self.get_document_count()

        if current_count == 0:
            console.print("[yellow]⚠️  No documents found in vector store.[/yellow]")
            return

        console.print(
            f"[yellow]⚠️  Warning: This will delete all {current_count} documents from the vector store![/yellow]"
        )

        # Confirm deletion
        confirm = Prompt.ask(
            "[bold red]Type 'CONFIRM' to proceed with deletion[/bold red]",
            default="",
        )

        if confirm.strip() != "CONFIRM":
            console.print("[yellow]❌ Deletion cancelled.[/yellow]")
            return

        try:
            console.print("[bold red]🗑️  Deleting all documents...[/bold red]")
            success = await self.rag.delete_all_documents("CONFIRM")

            if success:
                console.print(
                    "[bold green]✅ All documents deleted successfully![/bold green]"
                )
            else:
                console.print("[bold red]❌ Failed to delete documents.[/bold red]")

        except Exception as e:
            console.print(f"[bold red]❌ Error deleting documents: {e}[/bold red]")

    async def delete_all_web_search_results(self):
        """Delete all web search results from the vector store"""
        # Show current document count
        current_count = await self.get_document_count()

        if current_count == 0:
            console.print("[yellow]⚠️  No documents found in vector store.[/yellow]")
            return

        console.print(
            f"[yellow]⚠️  Warning: This will delete all {current_count} search results from the vector store![/yellow]"
        )

        # Confirm deletion
        confirm = Prompt.ask(
            "[bold red]Type 'CONFIRM' to proceed with deletion[/bold red]",
            default="",
        )

        if confirm.strip() != "CONFIRM":
            console.print("[yellow]❌ Deletion cancelled.[/yellow]")
            return

        try:
            console.print("[bold red]🗑️  Deleting all documents...[/bold red]")
            success = await self.rag.delete_all_documents("CONFIRM")

            if success:
                console.print(
                    "[bold green]✅ All web search results deleted successfully![/bold green]"
                )
            else:
                console.print(
                    "[bold red]❌ Failed to delete web search results.[/bold red]"
                )

        except Exception as e:
            console.print(
                f"[bold red]❌ Error deleting web search results: {e}[/bold red]"
            )

    async def interactive_menu(self):
        """Show interactive menu for additional options"""
        while True:
            console.print("\n[bold cyan]📋 Additional Options[/bold cyan]")
            console.print("1. Show engine status")
            console.print("2. Clear memory cache")
            console.print("3. Re-ingest documents")
            console.print("4. Delete all documents")
            console.print("5. Delete all search results")
            console.print("6. Return to chat")

            choice = Prompt.ask(
                "Select an option",
                choices=["1", "2", "3", "4", "5"],
                default="5",
            )

            if choice == "1":
                await self.show_engine_status()
            elif choice == "2":
                await self.clear_cache()
            elif choice == "3":
                await self.ingest_documents(force_ingest=True)
            elif choice == "4":
                await self.delete_all_documents()
            elif choice == "5":
                await self.delete_all_web_search_results()
            elif choice == "6":
                break


@app.command()
def chat(
    docs_path: str = typer.Option("./docs", help="Path to documents directory"),
    ingest: bool = typer.Option(False, help="Ingest documents before starting chat"),
    force_ingest: bool = typer.Option(
        False, help="Force re-ingestion even if documents exist"
    ),
):
    """Run the interactive RAG chat application"""

    async def app_main():
        chat = InteractiveRAGChat(docs_path)

        # Welcome message
        console.print(
            Panel.fit(
                "[bold cyan]🤖 Welcome to Reflexion RAG![/bold cyan]\n"
                "[dim]Documents: {docs_path}[/dim]".format(docs_path=docs_path)
            )
        )

        # Check if documents exist
        doc_count = await chat.get_document_count()

        if force_ingest or ingest or doc_count == 0:
            if doc_count == 0:
                console.print("[yellow]⚠️  No documents found in vector store.[/yellow]")
            else:
                console.print(
                    f"[yellow]📚 Current documents in store: {doc_count}[/yellow]"
                )
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

        # Show initial engine status
        await chat.show_engine_status()

        # Main chat loop
        console.print(
            "\n[bold green]💬 Chat started! Type your questions below.[/bold green]"
        )
        console.print(
            "[dim]Commands: 'exit' to quit, 'menu' for options, 'status' for engine info[/dim]"
        )

        while True:
            try:
                question = Prompt.ask("\n[bold blue]❓ Your question[/bold blue]")

                if question.strip().lower() in ["exit", "quit", "q"]:
                    console.print("[bold red]👋 Goodbye![/bold red]")
                    break
                elif question.strip().lower() == "menu":
                    await chat.interactive_menu()
                    continue
                elif question.strip().lower() == "status":
                    await chat.show_engine_status()
                    continue
                elif not question.strip():
                    console.print("[yellow]⚠️  Please enter a question.[/yellow]")
                    continue

                await chat.process_query_with_thinking(question)

            except KeyboardInterrupt:
                console.print(
                    "\n[yellow]⚠️  Interrupted. Type 'exit' to quit properly.[/yellow]"
                )
            except Exception as e:
                console.print(f"\n[red]❌ Unexpected error: {e}[/red]")

    asyncio.run(app_main())


@app.command()
def ingest(
    docs_path: str = typer.Option("./docs", help="Path to documents directory"),
):
    """Ingest documents into the vector store"""

    async def ingest_main():
        chat = InteractiveRAGChat(docs_path)
        console.print("[bold blue]📥 Document Ingestion Mode[/bold blue]")

        # Show current document count
        current_count = await chat.get_document_count()
        console.print(f"[dim]Current documents in store: {current_count}[/dim]")

        success = await chat.ingest_documents()
        if success:
            # Show new count
            new_count = await chat.get_document_count()
            console.print(
                f"[bold green]✅ Ingestion completed successfully! Total documents: {new_count}[/bold green]"
            )
        else:
            console.print("[bold red]❌ Ingestion failed![/bold red]")

    asyncio.run(ingest_main())


@app.command()
def deletedocs():
    """Delete all documents from the vector store"""

    async def delete_main():
        chat = InteractiveRAGChat("./docs")  # Path doesn't matter for delete
        console.print("[bold red]🗑️  Document Deletion Mode[/bold red]")
        await chat.delete_all_documents()

    asyncio.run(delete_main())


@app.command()
def deleteweb():
    """Delete all web search results from the vector store"""

    async def delete_main():
        chat = InteractiveRAGChat("./docs")  # Path doesn't matter for delete
        console.print("[bold red]🗑️  Web Search Deletion Mode[/bold red]")
        await chat.delete_all_web_search_results()

    asyncio.run(delete_main())


@app.command()
def config():
    """Show current configuration"""
    console.print("[bold cyan]⚙️  Reflexion RAG Configuration[/bold cyan]")

    config_table = Table(title="Settings", show_header=True)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    # Reflexion settings
    config_table.add_row("Max Reflexion Cycles", str(settings.max_reflexion_cycles))
    config_table.add_row("Confidence Threshold", str(settings.confidence_threshold))
    config_table.add_row("Initial Retrieval K", str(settings.initial_retrieval_k))
    config_table.add_row("Reflexion Retrieval K", str(settings.reflexion_retrieval_k))

    # Memory settings
    config_table.add_row("Memory Cache Enabled", str(settings.enable_memory_cache))
    config_table.add_row("Max Cache Size", str(settings.max_cache_size))

    # Models
    config_table.add_row("Generation Model", settings.llm_model)
    config_table.add_row("Evaluation Model", settings.evaluation_model)
    config_table.add_row("Summary Model", settings.summary_model)

    # Vector store settings
    config_table.add_row("Embedding Model", settings.embedding_model)
    config_table.add_row("Chunk Size", str(settings.chunk_size))
    config_table.add_row("Chunk Overlap", str(settings.chunk_overlap))

    console.print(config_table)


if __name__ == "__main__":
    app()
