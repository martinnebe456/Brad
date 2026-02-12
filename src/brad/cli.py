from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from brad.audio.ffmpeg import FfmpegError
from brad.config import get_settings
from brad.doctor import run_doctor
from brad.services import BradService

app = typer.Typer(help="Brad - your local AI meeting assistant")
console = Console()


@app.command()
def doctor() -> None:
    """Check local runtime prerequisites."""

    settings = get_settings()
    checks = run_doctor(settings)

    table = Table(title="Brad doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    failed = False
    for check in checks:
        status = check.status.upper()
        color = {"ok": "green", "warn": "yellow", "fail": "red"}.get(check.status, "white")
        table.add_row(check.name, f"[{color}]{status}[/{color}]", check.detail)
        if check.status == "fail":
            failed = True

    console.print(table)
    if failed:
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    file_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    model: str = typer.Option("small", "--model", help="small|medium|large"),
    language: str = typer.Option("auto", "--language", help="auto|cs|en"),
    vad: str = typer.Option("off", "--vad", help="on|off"),
) -> None:
    """Transcribe local audio and store transcript in SQLite."""

    use_vad = vad.lower().strip() == "on"
    service = BradService()
    try:
        outcome = service.transcribe_file(
            file_path,
            model_name=model,
            language=language,
            use_vad=use_vad,
        )
    except FfmpegError as exc:
        console.print(f"[red]ffmpeg error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[red]transcribe failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print(f"[green]Meeting ID:[/green] {outcome.meeting_id}")
    console.print(f"[green]Detected language:[/green] {outcome.language}")
    console.print(f"[green]Segments:[/green] {outcome.segment_count}")
    console.print("[green]Exports:[/green]")
    for fmt, path in outcome.export_paths.items():
        console.print(f"- {fmt}: {path}")


@app.command()
def summarize(
    target: str = typer.Argument(..., help="meeting_id or transcript file path"),
    template: str = typer.Option("general", "--template", help="general|sales|engineering"),
    llm_model: Path | None = typer.Option(None, "--llm-model", help="Path to GGUF file"),
) -> None:
    """Summarize a stored meeting or a transcript file."""

    service = BradService()
    try:
        outcome = service.summarize_target(
            target,
            template_name=template,
            llm_model=llm_model,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[red]summarize failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    if outcome.meeting_id is not None:
        console.print(f"[green]Summary stored for meeting:[/green] {outcome.meeting_id}")
    else:
        console.print("[yellow]Summary generated from transcript file (not stored in DB).[/yellow]")
    console.print(f"[green]Method:[/green] {outcome.summary.method}")
    console.print("")
    console.print(outcome.summary.text)


@app.command("export")
def export_cmd(
    meeting_id: int = typer.Argument(..., help="Meeting ID"),
    export_format: str = typer.Option("md", "--format", help="md|srt|json"),
) -> None:
    """Export a meeting transcript/summary artifact."""

    service = BradService()
    try:
        output = service.export_meeting(meeting_id, export_format)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]export failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    console.print(f"[green]Export written:[/green] {output}")


@app.command()
def search(
    query: str = typer.Argument(..., help="FTS query"),
    meeting: int | None = typer.Option(None, "--meeting", help="Filter by meeting id"),
) -> None:
    """Search transcripts using SQLite FTS5."""

    service = BradService()
    hits = service.search(query, meeting_id=meeting)
    if not hits:
        console.print("[yellow]No matches found.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="Search results")
    table.add_column("Meeting")
    table.add_column("Segment")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Snippet")
    for hit in hits:
        table.add_row(
            str(hit.meeting_id),
            str(hit.segment_id),
            f"{hit.start:.2f}",
            f"{hit.end:.2f}",
            hit.snippet,
        )
    console.print(table)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
) -> None:
    """Launch local Gradio UI."""

    from brad.ui.gradio_app import build_app

    demo = build_app()
    demo.launch(server_name=host, server_port=port, share=False, show_error=True)


if __name__ == "__main__":
    app()
