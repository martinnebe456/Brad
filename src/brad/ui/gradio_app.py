from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from brad.services import BradService


def build_app() -> gr.Blocks:
    service = BradService()

    def do_transcribe(
        audio_path: str | None,
        model: str,
        language: str,
        use_vad: bool,
    ):
        if not audio_path:
            return "", "Upload an audio file first.", ""
        try:
            outcome = service.transcribe_file(
                Path(audio_path),
                model_name=model,
                language=language,
                use_vad=use_vad,
            )
        except Exception as exc:  # pragma: no cover - UI path
            return "", f"Transcription failed: {exc}", ""

        meeting_id = outcome.meeting_id
        segments = service.db.get_segments(meeting_id)[:20]
        preview = "\n".join(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}" for seg in segments)
        exports = "\n".join(f"{fmt}: {path}" for fmt, path in outcome.export_paths.items())
        return str(meeting_id), preview or "(no segments)", exports

    def do_summarize(meeting_id_text: str, template: str, llm_model_path: str):
        if not meeting_id_text.strip():
            return "Provide a meeting ID."
        llm_model = Path(llm_model_path).expanduser() if llm_model_path.strip() else None
        try:
            outcome = service.summarize_target(
                meeting_id_text.strip(),
                template_name=template,
                llm_model=llm_model,
            )
            return outcome.summary.text
        except Exception as exc:  # pragma: no cover - UI path
            return f"Summarization failed: {exc}"

    def do_search(query: str, meeting_id_text: str):
        if not query.strip():
            return []
        meeting_id = int(meeting_id_text) if meeting_id_text.strip().isdigit() else None
        hits = service.search(query.strip(), meeting_id=meeting_id, limit=30)
        return [
            [hit.meeting_id, hit.segment_id, round(hit.start, 2), round(hit.end, 2), hit.snippet]
            for hit in hits
        ]

    def do_export(meeting_id_text: str, export_format: str):
        if not meeting_id_text.strip().isdigit():
            return "Provide a numeric meeting ID."
        try:
            path = service.export_meeting(int(meeting_id_text), export_format)
            return str(path)
        except Exception as exc:  # pragma: no cover - UI path
            return f"Export failed: {exc}"

    with gr.Blocks(title="Brad - local meeting assistant", analytics_enabled=False) as demo:
        gr.Markdown("# Brad - local meeting assistant")
        gr.Markdown("Upload -> transcribe -> summarize -> search -> export")

        with gr.Tab("Transcribe"):
            audio_input = gr.File(label="Audio file", type="filepath")
            model = gr.Dropdown(choices=["small", "medium", "large"], value="small", label="ASR model")
            language = gr.Dropdown(choices=["auto", "cs", "en"], value="auto", label="Language")
            use_vad = gr.Checkbox(value=False, label="Use Silero VAD")
            transcribe_button = gr.Button("Transcribe")
            meeting_id_box = gr.Textbox(label="Meeting ID")
            transcript_preview = gr.Textbox(label="Transcript preview", lines=12)
            export_paths = gr.Textbox(label="Generated exports", lines=4)

            transcribe_button.click(
                do_transcribe,
                inputs=[audio_input, model, language, use_vad],
                outputs=[meeting_id_box, transcript_preview, export_paths],
            )

        with gr.Tab("Summarize"):
            summarize_meeting_id = gr.Textbox(label="Meeting ID")
            template = gr.Dropdown(
                choices=["general", "sales", "engineering"],
                value="general",
                label="Template",
            )
            llm_model_path = gr.Textbox(
                label="GGUF model path (optional)",
                placeholder="~/.brad/models/llm/your-model.gguf",
            )
            summarize_button = gr.Button("Summarize")
            summary_output = gr.Textbox(label="Summary", lines=14)
            summarize_button.click(
                do_summarize,
                inputs=[summarize_meeting_id, template, llm_model_path],
                outputs=[summary_output],
            )

        with gr.Tab("Search"):
            query = gr.Textbox(label="Query")
            search_meeting = gr.Textbox(label="Meeting ID (optional)")
            search_button = gr.Button("Search")
            results = gr.Dataframe(
                headers=["meeting_id", "segment_id", "start", "end", "snippet"],
                datatype=["number", "number", "number", "number", "str"],
                row_count=(0, "dynamic"),
            )
            search_button.click(do_search, inputs=[query, search_meeting], outputs=[results])

        with gr.Tab("Export"):
            export_meeting = gr.Textbox(label="Meeting ID")
            export_format = gr.Dropdown(choices=["md", "srt", "json"], value="md", label="Format")
            export_button = gr.Button("Export")
            export_output = gr.Textbox(label="Output path")
            export_button.click(
                do_export,
                inputs=[export_meeting, export_format],
                outputs=[export_output],
            )

    return demo
