"""Enhanced input handler with multi-line support."""

from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

console = Console()


class MultiLineInput:
    """Handles multi-line user input for chat queries"""

    def __init__(self):
        self.last_input: Optional[str] = None
        self.multiline_mode = False

    def _print_hint(self) -> None:
        """Print input hints"""
        hints = Text()
        hints.append("Commands: ", style="dim")
        hints.append("exit", style="cyan")
        hints.append(" | ", style="dim")
        hints.append("menu", style="cyan")
        hints.append(" | ", style="dim")
        hints.append("status", style="cyan")
        hints.append(" | ", style="dim")
        hints.append("!!", style="yellow")
        hints.append(" (repeat) | ", style="dim")
        hints.append("\\m", style="yellow")
        hints.append(" (multiline)", style="dim")
        console.print(hints)

    def get_input(self, prompt: str = "Your question") -> Optional[str]:
        """
        Get user input with multi-line support.

        Modes:
        - Normal: Single Enter submits
        - Multi-line: Double Enter (empty line) submits
        - Special: !! repeats last input, \\m enters multiline mode

        Returns:
            User input string, or None if cancelled
        """
        while True:
            self._print_hint()

            try:
                # Get initial input
                raw_input = Prompt.ask(f"\n[bold blue]❓ {prompt}[/bold blue]")

                # Handle special commands
                if raw_input.strip().lower() in ["exit", "quit", "q"]:
                    return "exit"

                if raw_input.strip() == "!!":
                    if self.last_input:
                        console.print(
                            f"[dim]Repeating: {self.last_input[:50]}...[/dim]"
                        )
                        return self.last_input
                    else:
                        console.print("[yellow]No previous input to repeat.[/yellow]")
                        continue  # loop back to prompt

                if raw_input.strip().lower() == "\\m":
                    # Enter multiline mode
                    return self._get_multiline_input(prompt)

                if raw_input.strip().lower() == "\\e":
                    # Cancel - get new input
                    console.print("[dim]Input cancelled.[/dim]")
                    continue  # loop back to prompt

                # Check if user wants to continue with more lines
                # (empty input followed by content suggests multiline)
                if raw_input.strip() == "":
                    console.print(
                        "[dim]Empty input. Type your question or 'exit'.[/dim]"
                    )
                    continue  # loop back to prompt

                # Store and return
                self.last_input = raw_input.strip()
                return raw_input.strip()

            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Input cancelled.[/dim]")
                return None

    def _get_multiline_input(self, prompt: str) -> Optional[str]:
        """Get multi-line input from user"""
        console.print("[bold cyan]📝 Multi-line Mode[/bold cyan]")
        console.print(
            "[dim]Type your input. Press Enter twice (empty line) to submit.[/dim]"
        )
        console.print("[dim]Type \\e to cancel.[/dim]")
        console.print()

        while True:
            lines = []
            empty_line_count = 0

            while True:
                try:
                    # Show line number
                    line_num = len(lines) + 1
                    line = Prompt.ask(f"  [dim]{line_num:2d}|[/dim]")

                    if line.strip().lower() == "\\e":
                        console.print("[dim]Multi-line input cancelled.[/dim]")
                        # Fall back to normal single-line input
                        return self.get_input(prompt)

                    if line.strip() == "":
                        empty_line_count += 1
                        if empty_line_count >= 2 and len(lines) > 0:
                            # Double enter with content - submit
                            break
                    else:
                        empty_line_count = 0
                        lines.append(line)

                except (KeyboardInterrupt, EOFError):
                    console.print("\n[dim]Multi-line input cancelled.[/dim]")
                    return self.get_input(prompt)

            if not lines:
                console.print("[dim]No input provided.[/dim]")
                # Re-enter multiline mode rather than recursing into get_input
                continue

            result = "\n".join(lines)

            # Show preview
            console.print()
            console.print("[dim]━━━ Input Preview ━━━[/dim]")
            preview = result[:200] + "..." if len(result) > 200 else result
            console.print(f"[dim]{preview}[/dim]")
            console.print("[dim]━━━━━━━━━━━━━━━━━━━[/dim]")

            self.last_input = result
            return result

    def get_last_input(self) -> Optional[str]:
        """Get the last submitted input"""
        return self.last_input


class EnhancedInputHandler:
    """Enhanced input handler with history and shortcuts"""

    def __init__(self, max_history: int = 50):
        self.multiline = MultiLineInput()
        self.history: list[str] = []
        self.max_history = max_history

    def add_to_history(self, input_str: str) -> None:
        """Add input to history"""
        if (
            input_str
            and input_str.strip()
            and input_str not in ["exit", "menu", "status"]
        ):
            # Avoid duplicates
            if input_str in self.history:
                self.history.remove(input_str)
            self.history.append(input_str)

            # Trim history
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]

    def get_input(self, prompt: str = "Your question") -> Optional[str]:
        """Get user input with all enhancements"""
        result = self.multiline.get_input(prompt)

        if result and result != "exit":
            self.add_to_history(result)

        return result

    def get_history(self, limit: int = 10) -> list[str]:
        """Get recent input history"""
        return self.history[-limit:]

    def print_history(self, limit: int = 10) -> None:
        """Print recent input history"""
        if not self.history:
            console.print("[dim]No input history.[/dim]")
            return

        console.print(f"\n[bold cyan]📜 Input History (last {limit})[/bold cyan]")
        recent = self.get_history(limit)

        for i, item in enumerate(recent, 1):
            preview = item[:50] + "..." if len(item) > 50 else item
            console.print(f"  [dim]{i}.[/dim] {preview}")

    def repeat_last(self) -> Optional[str]:
        """Repeat the last input"""
        if self.history:
            return self.history[-1]
        return None


def create_input_handler() -> EnhancedInputHandler:
    """Create an enhanced input handler instance"""
    return EnhancedInputHandler()


if __name__ == "__main__":
    # Test the input handler
    handler = create_input_handler()

    console.print("[bold]Testing Enhanced Input Handler[/bold]")
    console.print("Type 'exit' to quit\n")

    while True:
        result = handler.get_input("Test input")

        if result is None:
            continue
        elif result == "exit":
            console.print("[bold red]Goodbye![/bold red]")
            break
        else:
            console.print("\n[green]You entered:[/green]")
            console.print(f"  {result[:100]}...")
            console.print()
