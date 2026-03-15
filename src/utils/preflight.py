"""Pre-flight checks for validating API credentials before starting the CLI."""

from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from src.config.settings import settings

console = Console()


@dataclass
class CredentialCheck:
    """Result of a credential check"""

    name: str
    required: bool
    present: bool
    source: str  # 'env', 'file', 'default', 'missing'
    note: Optional[str] = None


class PreflightChecker:
    """Validates API credentials before starting the application"""

    REQUIRED_CREDENTIALS = [
        ("github_token", "GitHub Token", "Required for LLM and embeddings"),
        ("surrealdb_url", "SurrealDB URL", "Database connection"),
        ("surrealdb_user", "SurrealDB Username", "Database authentication"),
        ("surrealdb_pass", "SurrealDB Password", "Database authentication"),
    ]

    OPTIONAL_CREDENTIALS = [
        ("google_api_key", "Google API Key", "Required for web search"),
        ("google_cse_id", "Google CSE ID", "Required for web search"),
    ]

    def __init__(self):
        self.results: List[CredentialCheck] = []

    def _check_credential(
        self, attr_name: str, display_name: str, note: str, required: bool
    ) -> CredentialCheck:
        """Check if a credential is present in settings"""
        # Get the value from the already-loaded settings object
        value = getattr(settings, attr_name, None)

        # Check if value is present and not a placeholder
        is_present = False
        source = "missing"

        if value:
            # Check if it's a placeholder/default value
            placeholder_patterns = [
                "your_",  # your_username, your_password, etc.
                "ghp_",  # GitHub token prefix (but this is actually valid)
            ]

            # For GitHub token, ghp_ prefix is valid
            if attr_name == "github_token" and value.startswith(
                ("ghp_", "github_pat_")
            ):
                is_present = True
                source = "env"
            elif value and not any(value.lower().startswith(p) for p in ["your_"]):
                is_present = True
                source = "env"
            elif value and not str(value).startswith("your_"):
                is_present = True
                source = "env"

        return CredentialCheck(
            name=display_name,
            required=required,
            present=is_present,
            source=source,
            note=note,
        )

    def run_checks(self) -> bool:
        """Run all credential checks and return True if all required are present"""
        self.results = []
        all_required_present = True

        # Check required credentials
        for attr_name, display_name, note in self.REQUIRED_CREDENTIALS:
            result = self._check_credential(
                attr_name, display_name, note, required=True
            )
            self.results.append(result)

            if not result.present:
                all_required_present = False

        # Check optional credentials
        for attr_name, display_name, note in self.OPTIONAL_CREDENTIALS:
            result = self._check_credential(
                attr_name, display_name, note, required=False
            )
            self.results.append(result)

        return all_required_present

    def get_feature_status(self) -> dict:
        """Get status of optional features based on credentials"""
        has_google_api = any(
            r.name == "Google API Key" and r.present for r in self.results
        )
        has_google_cse = any(
            r.name == "Google CSE ID" and r.present for r in self.results
        )
        has_github_token = any(
            r.name == "GitHub Token" and r.present for r in self.results
        )
        has_db = all(
            r.present for r in self.results if r.required and "SurrealDB" in r.name
        )

        return {
            "web_search": has_google_api and has_google_cse,
            "llm": has_github_token,
            "database": has_db,
        }

    def print_report(self) -> None:
        """Print a formatted report of credential status"""
        console.print()
        console.print("[bold cyan]🔍 Pre-flight Credential Check[/bold cyan]")
        console.print("━" * 50)

        # Required credentials table
        console.print("\n[bold]Required Credentials:[/bold]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("Credential", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Source", style="dim")
        table.add_column("Note", style="dim")

        for result in self.results:
            if result.required:
                status = (
                    "[green]✓ Present[/green]"
                    if result.present
                    else "[red]✗ Missing[/red]"
                )
                source = result.source if result.present else "—"
                table.add_row(result.name, status, source, result.note or "")

        console.print(table)

        # Optional credentials table
        console.print("\n[bold]Optional Credentials:[/bold]")
        opt_table = Table(show_header=True, header_style="dim")
        opt_table.add_column("Credential", style="cyan")
        opt_table.add_column("Status", justify="center")
        opt_table.add_column("Source", style="dim")
        opt_table.add_column("Note", style="dim")

        for result in self.results:
            if not result.required:
                status = (
                    "[green]✓ Present[/green]"
                    if result.present
                    else "[dim]○ Not set[/dim]"
                )
                source = result.source if result.present else "—"
                opt_table.add_row(result.name, status, source, result.note or "")

        console.print(opt_table)

        # Feature status
        features = self.get_feature_status()
        console.print("\n[bold]Feature Status:[/bold]")

        feature_status = []
        if features["web_search"]:
            feature_status.append("[green]🌐 Web Search: Enabled[/green]")
        else:
            feature_status.append(
                "[dim]🌐 Web Search: Disabled[/dim] (requires Google API)"
            )

        if features["llm"]:
            feature_status.append("[green]🤖 LLM: Enabled[/green]")
        else:
            feature_status.append("[red]🤖 LLM: Disabled[/red]")

        if features["database"]:
            feature_status.append("[green]💾 Database: Connected[/green]")
        else:
            feature_status.append("[red]💾 Database: Not configured[/red]")

        console.print("  " + " │ ".join(feature_status))

    def print_errors_and_exit(self) -> bool:
        """Print error messages for missing required credentials"""
        missing = [r for r in self.results if r.required and not r.present]

        if missing:
            console.print()
            console.print("[bold red]❌ Missing Required Credentials[/bold red]")
            console.print("━" * 50)
            console.print()

            for result in missing:
                console.print(f"[red]• {result.name}:[/red] {result.note}")

            console.print()
            console.print(
                "[yellow]Please set these credentials in your .env file:[/yellow]"
            )
            console.print("[dim]  .env file location: project root directory[/dim]")
            console.print()
            console.print("[dim]Example .env file:[/dim]")
            console.print("[dim]  GITHUB_TOKEN=ghp_your_token_here[/dim]")
            console.print(
                "[dim]  SURREALDB_URL=wss://your-instance.surreal.cloud[/dim]"
            )
            console.print("[dim]  SURREALDB_USER=your_username[/dim]")
            console.print("[dim]  SURREALDB_PASS=your_password[/dim]")
            console.print()

        return len(missing) == 0


def run_preflight_checks(exit_on_error: bool = True) -> bool:
    """
    Run pre-flight checks and optionally exit on error.

    Args:
        exit_on_error: If True, exit program on missing required credentials

    Returns:
        True if all required credentials are present
    """
    checker = PreflightChecker()
    success = checker.run_checks()
    checker.print_report()

    if not success and exit_on_error:
        checker.print_errors_and_exit()
        import sys

        sys.exit(1)

    return success


if __name__ == "__main__":
    run_preflight_checks()
