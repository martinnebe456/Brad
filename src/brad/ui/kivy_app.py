from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from brad.doctor import DoctorCheck, run_doctor
from brad.services import BradService, SummaryOutcome, TranscriptionOutcome
from brad.storage.models import SearchHit


def launch_kivy_app() -> None:
    """Launch experimental Kivy desktop UI."""

    try:
        from kivy.app import App
        from kivy.clock import Clock
        from kivy.core.window import Window
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from kivy.uix.gridlayout import GridLayout
        from kivy.uix.label import Label
        from kivy.uix.spinner import Spinner
        from kivy.uix.textinput import TextInput
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Kivy is not installed. Install optional dependency with: "
            "pip install -e '.[ui-kivy]'"
        ) from exc

    class BradKivyApp(App):  # type: ignore[misc]
        page_meta = {
            "transcribe": "Transcribe",
            "summarize": "Summarize",
            "search": "Search",
            "export": "Export",
            "health": "Health",
        }

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.service = BradService()
            self.executor = ThreadPoolExecutor(max_workers=2)
            self._busy = False
            self._action_buttons: list[Button] = []
            self._current_page: str | None = None
            self.last_meeting_id: int | None = None

            self.nav_buttons: dict[str, Button] = {}
            self.pages: dict[str, BoxLayout] = {}

            self.page_title: Label
            self.last_meeting_label: Label
            self.status_label: Label
            self.page_host: BoxLayout

            self.transcribe_audio_input: TextInput
            self.transcribe_model_spinner: Spinner
            self.transcribe_language_spinner: Spinner
            self.transcribe_vad_spinner: Spinner
            self.transcribe_output: TextInput

            self.summary_target_input: TextInput
            self.summary_template_spinner: Spinner
            self.summary_llm_input: TextInput
            self.summary_output: TextInput

            self.search_query_input: TextInput
            self.search_meeting_input: TextInput
            self.search_output: TextInput

            self.export_meeting_input: TextInput
            self.export_format_spinner: Spinner
            self.export_output: TextInput

            self.doctor_output: TextInput

        def build(self) -> BoxLayout:
            Window.minimum_width = 980
            Window.minimum_height = 640
            Window.size = (1200, 760)

            root = BoxLayout(orientation="horizontal", padding=10, spacing=10)

            nav = BoxLayout(
                orientation="vertical",
                spacing=6,
                size_hint=(None, 1),
                width=180,
            )
            nav.add_widget(self._make_label("Brad Workflows", bold=True))
            for key, title in self.page_meta.items():
                button = self._make_button(
                    title,
                    on_click=lambda _btn, page_key=key: self._show_page(page_key),
                    height=40,
                )
                nav.add_widget(button)
                self.nav_buttons[key] = button
            nav.add_widget(self._make_button("Use Last For Summary", self._on_use_last_for_summary))
            nav.add_widget(self._make_button("Use Last For Export", self._on_use_last_for_export))
            nav.add_widget(self._make_button("Run Doctor", self._on_doctor))
            root.add_widget(nav)

            main = BoxLayout(orientation="vertical", spacing=8)
            header = BoxLayout(orientation="horizontal", size_hint=(1, None), height=44)
            self.page_title = self._make_label("Transcribe", bold=True)
            self.last_meeting_label = self._make_label("Last meeting: -")
            header.add_widget(self.page_title)
            header.add_widget(self.last_meeting_label)
            main.add_widget(header)

            self.page_host = BoxLayout(orientation="vertical")
            main.add_widget(self.page_host)

            self.status_label = self._make_label("Ready.", height=30)
            main.add_widget(self.status_label)

            root.add_widget(main)

            self.pages = {
                "transcribe": self._build_transcribe_page(),
                "summarize": self._build_summarize_page(),
                "search": self._build_search_page(),
                "export": self._build_export_page(),
                "health": self._build_health_page(),
            }
            self._show_page("transcribe")
            Clock.schedule_once(lambda _dt: self._on_doctor(None), 0.1)
            return root

        def on_stop(self) -> None:
            self.executor.shutdown(wait=False, cancel_futures=True)

        def _make_label(
            self,
            text: str,
            *,
            bold: bool = False,
            height: int = 28,
        ) -> Label:
            label = Label(
                text=f"[b]{text}[/b]" if bold else text,
                markup=bold,
                halign="left",
                valign="middle",
                size_hint=(1, None),
                height=height,
                color=(0.13, 0.14, 0.17, 1),
            )
            label.bind(size=lambda instance, _size: setattr(instance, "text_size", instance.size))
            return label

        def _make_button(
            self,
            text: str,
            on_click: Callable[[Button], None],
            *,
            height: int = 36,
        ) -> Button:
            button = Button(
                text=text,
                size_hint=(1, None),
                height=height,
                background_normal="",
                background_color=(0.89, 0.9, 0.93, 1),
                color=(0.13, 0.14, 0.17, 1),
            )
            button.bind(on_release=on_click)
            return button

        def _make_text_input(
            self,
            *,
            multiline: bool = False,
            readonly: bool = False,
            height: int = 34,
        ) -> TextInput:
            return TextInput(
                multiline=multiline,
                readonly=readonly,
                size_hint=(1, None) if not multiline else (1, 1),
                height=height if not multiline else 200,
                background_color=(0.97, 0.97, 0.98, 1),
                foreground_color=(0.13, 0.14, 0.17, 1),
                cursor_color=(0.13, 0.14, 0.17, 1),
            )

        def _build_transcribe_page(self) -> BoxLayout:
            page = BoxLayout(orientation="vertical", spacing=8)
            controls = GridLayout(cols=2, spacing=6, size_hint=(1, None))
            controls.bind(minimum_height=controls.setter("height"))

            self.transcribe_audio_input = self._make_text_input()
            self.transcribe_model_spinner = Spinner(text="small", values=("small", "medium", "large"))
            self.transcribe_language_spinner = Spinner(text="auto", values=("auto", "cs", "en"))
            self.transcribe_vad_spinner = Spinner(text="off", values=("off", "on"))

            controls.add_widget(self._make_label("Audio path", height=30))
            controls.add_widget(self.transcribe_audio_input)
            controls.add_widget(self._make_label("Model", height=30))
            controls.add_widget(self.transcribe_model_spinner)
            controls.add_widget(self._make_label("Language", height=30))
            controls.add_widget(self.transcribe_language_spinner)
            controls.add_widget(self._make_label("VAD", height=30))
            controls.add_widget(self.transcribe_vad_spinner)

            page.add_widget(controls)
            run_button = self._make_button("Start Transcription", self._on_transcribe, height=40)
            self._action_buttons.append(run_button)
            page.add_widget(run_button)

            self.transcribe_output = self._make_text_input(multiline=True, readonly=True)
            page.add_widget(self.transcribe_output)
            return page

        def _build_summarize_page(self) -> BoxLayout:
            page = BoxLayout(orientation="vertical", spacing=8)
            controls = GridLayout(cols=2, spacing=6, size_hint=(1, None))
            controls.bind(minimum_height=controls.setter("height"))

            self.summary_target_input = self._make_text_input()
            self.summary_template_spinner = Spinner(
                text="general",
                values=("general", "sales", "engineering"),
            )
            self.summary_llm_input = self._make_text_input()

            controls.add_widget(self._make_label("Target (meeting ID or path)", height=30))
            controls.add_widget(self.summary_target_input)
            controls.add_widget(self._make_label("Template", height=30))
            controls.add_widget(self.summary_template_spinner)
            controls.add_widget(self._make_label("LLM model path (optional)", height=30))
            controls.add_widget(self.summary_llm_input)
            page.add_widget(controls)

            run_button = self._make_button("Generate Summary", self._on_summarize, height=40)
            self._action_buttons.append(run_button)
            page.add_widget(run_button)

            self.summary_output = self._make_text_input(multiline=True, readonly=True)
            page.add_widget(self.summary_output)
            return page

        def _build_search_page(self) -> BoxLayout:
            page = BoxLayout(orientation="vertical", spacing=8)
            controls = GridLayout(cols=2, spacing=6, size_hint=(1, None))
            controls.bind(minimum_height=controls.setter("height"))

            self.search_query_input = self._make_text_input()
            self.search_meeting_input = self._make_text_input()

            controls.add_widget(self._make_label("Query", height=30))
            controls.add_widget(self.search_query_input)
            controls.add_widget(self._make_label("Meeting ID (optional)", height=30))
            controls.add_widget(self.search_meeting_input)
            page.add_widget(controls)

            run_button = self._make_button("Search", self._on_search, height=40)
            self._action_buttons.append(run_button)
            page.add_widget(run_button)

            self.search_output = self._make_text_input(multiline=True, readonly=True)
            page.add_widget(self.search_output)
            return page

        def _build_export_page(self) -> BoxLayout:
            page = BoxLayout(orientation="vertical", spacing=8)
            controls = GridLayout(cols=2, spacing=6, size_hint=(1, None))
            controls.bind(minimum_height=controls.setter("height"))

            self.export_meeting_input = self._make_text_input()
            self.export_format_spinner = Spinner(text="md", values=("md", "srt", "json"))

            controls.add_widget(self._make_label("Meeting ID", height=30))
            controls.add_widget(self.export_meeting_input)
            controls.add_widget(self._make_label("Format", height=30))
            controls.add_widget(self.export_format_spinner)
            page.add_widget(controls)

            run_button = self._make_button("Export", self._on_export, height=40)
            self._action_buttons.append(run_button)
            page.add_widget(run_button)

            self.export_output = self._make_text_input(multiline=True, readonly=True)
            page.add_widget(self.export_output)
            return page

        def _build_health_page(self) -> BoxLayout:
            page = BoxLayout(orientation="vertical", spacing=8)
            run_button = self._make_button("Run Doctor", self._on_doctor, height=40)
            self._action_buttons.append(run_button)
            page.add_widget(run_button)
            self.doctor_output = self._make_text_input(multiline=True, readonly=True)
            page.add_widget(self.doctor_output)
            return page

        def _show_page(self, page_key: str) -> None:
            selected = self.pages.get(page_key)
            if selected is None:
                return
            self.page_host.clear_widgets()
            self.page_host.add_widget(selected)
            self._set_title(self.page_meta.get(page_key, "Brad"))

            for key, button in self.nav_buttons.items():
                button.background_color = (
                    (0.79, 0.82, 0.9, 1) if key == page_key else (0.89, 0.9, 0.93, 1)
                )

        def _set_title(self, title: str) -> None:
            self.page_title.text = f"[b]{title}[/b]"
            self.page_title.markup = True

        def _set_status(self, message: str) -> None:
            self.status_label.text = message

        def _set_last_meeting(self, meeting_id: int) -> None:
            self.last_meeting_id = meeting_id
            self.last_meeting_label.text = f"Last meeting: {meeting_id}"

        def _set_busy(self, busy: bool, status: str) -> None:
            self._busy = busy
            for button in self._action_buttons:
                button.disabled = busy
            self._set_status(status)

        def _run_async(
            self,
            busy_status: str,
            task_fn: Callable[[], object],
            on_success: Callable[[object], None],
        ) -> None:
            if self._busy:
                self._set_status("Another task is already running. Please wait.")
                return
            self._set_busy(True, busy_status)
            future = self.executor.submit(task_fn)
            future.add_done_callback(
                lambda done_future: Clock.schedule_once(
                    lambda _dt, result_future=done_future: self._finish_future(
                        result_future,
                        on_success,
                    )
                )
            )

        def _finish_future(
            self,
            future: Future[object],
            on_success: Callable[[object], None],
        ) -> None:
            try:
                result = future.result()
            except Exception as exc:
                self._set_busy(False, f"Error: {exc}")
                return

            try:
                on_success(result)
            except Exception as exc:
                self._set_busy(False, f"UI update error: {exc}")
                return

            self._set_busy(False, self.status_label.text)

        def _on_use_last_for_summary(self, _button: Button | None) -> None:
            if self.last_meeting_id is not None:
                self.summary_target_input.text = str(self.last_meeting_id)

        def _on_use_last_for_export(self, _button: Button | None) -> None:
            if self.last_meeting_id is not None:
                value = str(self.last_meeting_id)
                self.export_meeting_input.text = value
                self.search_meeting_input.text = value

        def _on_transcribe(self, _button: Button | None) -> None:
            raw_path = self.transcribe_audio_input.text.strip()
            if not raw_path:
                self._set_status("Select an audio file path first.")
                return
            audio_path = Path(raw_path).expanduser()
            if not audio_path.exists():
                self._set_status(f"Audio file not found: {audio_path}")
                return

            model = self.transcribe_model_spinner.text.strip()
            language = self.transcribe_language_spinner.text.strip()
            use_vad = self.transcribe_vad_spinner.text.strip().lower() == "on"

            def task() -> TranscriptionOutcome:
                return self.service.transcribe_file(
                    audio_path,
                    model_name=model,
                    language=language,
                    use_vad=use_vad,
                )

            def done(outcome: TranscriptionOutcome) -> None:
                self._set_last_meeting(outcome.meeting_id)
                meeting_text = str(outcome.meeting_id)
                self.summary_target_input.text = meeting_text
                self.search_meeting_input.text = meeting_text
                self.export_meeting_input.text = meeting_text

                segments = self.service.db.get_segments(outcome.meeting_id)
                preview = "\n".join(
                    f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}"
                    for segment in segments[:120]
                )
                exports = "\n".join(f"{fmt}: {path}" for fmt, path in outcome.export_paths.items())
                self.transcribe_output.text = (
                    f"{preview or '(no transcript segments)'}\n\n"
                    f"Exports:\n{exports or '(no exports)'}"
                )
                self._set_status(
                    f"Meeting {outcome.meeting_id} ready "
                    f"({outcome.segment_count} segments, {outcome.language})."
                )

            self._run_async("Transcribing audio...", task, done)

        def _on_summarize(self, _button: Button | None) -> None:
            target = self.summary_target_input.text.strip()
            if not target:
                self._set_status("Provide meeting ID or transcript file path.")
                return

            template = self.summary_template_spinner.text.strip()
            llm_raw = self.summary_llm_input.text.strip()
            llm_model = Path(llm_raw).expanduser() if llm_raw else None

            def task() -> SummaryOutcome:
                return self.service.summarize_target(
                    target,
                    template_name=template,
                    llm_model=llm_model,
                )

            def done(outcome: SummaryOutcome) -> None:
                if outcome.meeting_id is not None:
                    self._set_last_meeting(outcome.meeting_id)
                self.summary_output.text = outcome.summary.text
                self._set_status(f"Summary generated via {outcome.summary.method}.")

            self._run_async("Generating summary...", task, done)

        def _on_search(self, _button: Button | None) -> None:
            query = self.search_query_input.text.strip()
            if not query:
                self._set_status("Provide search query.")
                return

            meeting_text = self.search_meeting_input.text.strip()
            if meeting_text and not meeting_text.isdigit():
                self._set_status("Meeting ID must be numeric.")
                return
            meeting_id = int(meeting_text) if meeting_text else None

            def task() -> list[SearchHit]:
                return self.service.search(query, meeting_id=meeting_id, limit=100)

            def done(hits: list[SearchHit]) -> None:
                lines = [
                    (
                        f"meeting={hit.meeting_id} segment={hit.segment_id} "
                        f"[{hit.start:.2f}-{hit.end:.2f}] {hit.snippet}"
                    )
                    for hit in hits
                ]
                self.search_output.text = "\n".join(lines) or "(no matches)"
                self._set_status(f"Search done. {len(hits)} result(s).")

            self._run_async("Searching transcripts...", task, done)

        def _on_export(self, _button: Button | None) -> None:
            meeting_raw = self.export_meeting_input.text.strip()
            if not meeting_raw.isdigit():
                self._set_status("Meeting ID must be numeric.")
                return
            meeting_id = int(meeting_raw)
            export_format = self.export_format_spinner.text.strip()

            def task() -> Path:
                return self.service.export_meeting(meeting_id, export_format)

            def done(path: Path) -> None:
                self.export_output.text = str(path)
                self._set_status(f"Export created: {path}")

            self._run_async(f"Exporting meeting {meeting_id}...", task, done)

        def _on_doctor(self, _button: Button | None) -> None:
            def task() -> list[DoctorCheck]:
                return run_doctor(self.service.settings)

            def done(checks: list[DoctorCheck]) -> None:
                lines = [
                    f"{check.status.upper():<4} | {check.name}: {check.detail}"
                    for check in checks
                ]
                self.doctor_output.text = "\n".join(lines)
                failing = sum(1 for check in checks if check.status == "fail")
                warning = sum(1 for check in checks if check.status == "warn")
                self._set_status(
                    f"Doctor completed ({failing} fail, {warning} warn, {len(checks)} checks)."
                )

            self._run_async("Running environment checks...", task, done)

    BradKivyApp().run()
