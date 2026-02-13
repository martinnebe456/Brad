from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from brad.doctor import DoctorCheck, run_doctor
from brad.services import BradService, SummaryOutcome, TranscriptionOutcome
from brad.storage.models import SearchHit


class BradDesktopApp:
    """Native desktop UI for Brad workflows."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.service = BradService()
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.palette = {
            "bg": "#CFCED8",
            "shell": "#F4F4F6",
            "surface": "#FFFFFF",
            "surface_alt": "#ECECF1",
            "rail": "#E3E3E9",
            "sidebar": "#E9E9EE",
            "text": "#20232B",
            "muted": "#666B76",
            "border": "#D6D8E1",
            "accent": "#2E313A",
            "accent_active": "#1D1F26",
            "accent_disabled": "#AEB2BF",
            "selection": "#DDE2F2",
        }
        self.ui_font = "Segoe UI"

        self._busy = False
        self._action_buttons: list[ttk.Button] = []
        self.last_meeting_id: int | None = None

        self.status_var = tk.StringVar(value="Ready.")
        self.last_meeting_var = tk.StringVar(value="-")

        self.audio_path_var = tk.StringVar()
        self.model_var = tk.StringVar(value="small")
        self.language_var = tk.StringVar(value="auto")
        self.use_vad_var = tk.BooleanVar(value=False)

        self.summarize_target_var = tk.StringVar()
        self.template_var = tk.StringVar(value="general")
        self.llm_model_var = tk.StringVar()

        self.search_query_var = tk.StringVar()
        self.search_meeting_var = tk.StringVar()

        self.export_meeting_var = tk.StringVar()
        self.export_format_var = tk.StringVar(value="md")
        self.export_path_var = tk.StringVar()

        self.transcript_preview: ScrolledText
        self.export_paths_box: ScrolledText
        self.summary_output: ScrolledText
        self.search_tree: ttk.Treeview
        self.doctor_tree: ttk.Treeview
        self.page_title_var = tk.StringVar(value="Transcribe")
        self.page_subtitle_var = tk.StringVar(
            value="Convert audio into transcript segments and exports."
        )
        self.page_meta: dict[str, tuple[str, str]] = {
            "transcribe": (
                "Transcribe",
                "Convert audio into transcript segments and export files.",
            ),
            "summarize": (
                "Summarize",
                "Generate concise notes from a meeting ID or transcript file.",
            ),
            "search": (
                "Search",
                "Find key moments and snippets in stored meeting transcripts.",
            ),
            "export": (
                "Export",
                "Generate markdown, SRT or JSON artifacts for a selected meeting.",
            ),
            "health": (
                "Health",
                "Run environment checks and verify local runtime dependencies.",
            ),
        }
        self.page_icons: dict[str, str] = {
            "transcribe": "\U0001F3A4",
            "summarize": "\U0001F4DD",
            "search": "\U0001F50E",
            "export": "\U0001F4E4",
            "health": "\u2699",
        }
        self.action_icons: dict[str, str] = {
            "summary": "\u270D",
            "export": "\U0001F4BE",
            "doctor": "\u2699",
        }
        self.page_frames: dict[str, ttk.Frame] = {}
        self.nav_buttons: dict[str, ttk.Button] = {}
        self.rail_buttons: dict[str, ttk.Button] = {}

        self._configure_root()
        self._configure_style()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_root(self) -> None:
        self.root.title("Brad Desktop")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)
        self.root.configure(background=self.palette["bg"])
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=(self.ui_font, 10))

        style.configure("Root.TFrame", background=self.palette["bg"])
        style.configure("Shell.TFrame", background=self.palette["shell"])
        style.configure("Rail.TFrame", background=self.palette["rail"])
        style.configure("Sidebar.TFrame", background=self.palette["sidebar"])
        style.configure("Main.TFrame", background=self.palette["shell"])
        style.configure("App.TFrame", background=self.palette["shell"])
        style.configure(
            "HeaderCard.TFrame",
            background=self.palette["surface"],
            relief="solid",
            borderwidth=1,
        )
        style.configure("Status.TFrame", background=self.palette["surface_alt"])

        style.configure(
            "Card.TLabelframe",
            background=self.palette["surface"],
            relief="solid",
            borderwidth=1,
            padding=(12, 10),
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=self.palette["surface"],
            foreground=self.palette["muted"],
            font=(self.ui_font, 10, "bold"),
        )
        style.configure(
            "SidebarCard.TLabelframe",
            background=self.palette["sidebar"],
            relief="flat",
            borderwidth=0,
            padding=(8, 6),
        )
        style.configure(
            "SidebarCard.TLabelframe.Label",
            background=self.palette["sidebar"],
            foreground=self.palette["muted"],
            font=(self.ui_font, 9, "bold"),
        )

        style.configure(
            "TLabel",
            background=self.palette["shell"],
            foreground=self.palette["text"],
            font=(self.ui_font, 10),
        )
        style.configure(
            "RailLabel.TLabel",
            background=self.palette["rail"],
            foreground=self.palette["muted"],
            font=(self.ui_font, 9, "bold"),
        )
        style.configure(
            "SidebarTitle.TLabel",
            background=self.palette["sidebar"],
            foreground=self.palette["text"],
            font=(self.ui_font, 14, "bold"),
        )
        style.configure(
            "SidebarHint.TLabel",
            background=self.palette["sidebar"],
            foreground=self.palette["muted"],
            font=(self.ui_font, 9),
        )
        style.configure(
            "AppTitle.TLabel",
            background=self.palette["surface"],
            foreground=self.palette["text"],
            font=(self.ui_font, 15, "bold"),
        )
        style.configure(
            "AppSubtitle.TLabel",
            background=self.palette["surface"],
            foreground=self.palette["muted"],
            font=(self.ui_font, 10),
        )
        style.configure(
            "PageTitle.TLabel",
            background=self.palette["shell"],
            foreground=self.palette["text"],
            font=(self.ui_font, 16, "bold"),
        )
        style.configure(
            "PageSubtitle.TLabel",
            background=self.palette["shell"],
            foreground=self.palette["muted"],
            font=(self.ui_font, 10),
        )
        style.configure(
            "Badge.TLabel",
            background=self.palette["surface_alt"],
            foreground=self.palette["text"],
            font=(self.ui_font, 9, "bold"),
            padding=(8, 3),
        )
        style.configure(
            "StatusText.TLabel",
            background=self.palette["surface_alt"],
            foreground=self.palette["muted"],
            font=(self.ui_font, 9),
        )

        style.configure(
            "TButton",
            padding=(10, 6),
            foreground=self.palette["text"],
            background=self.palette["surface_alt"],
            borderwidth=1,
            relief="flat",
            font=(self.ui_font, 10),
        )
        style.map(
            "TButton",
            background=[("active", "#E3E3EB"), ("pressed", "#D9D9E3")],
        )
        style.configure(
            "Primary.TButton",
            padding=(12, 7),
            foreground="#FFFFFF",
            background=self.palette["accent"],
            borderwidth=0,
            relief="flat",
            font=(self.ui_font, 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[
                ("disabled", self.palette["accent_disabled"]),
                ("active", self.palette["accent_active"]),
                ("pressed", "#15171E"),
            ],
            foreground=[("disabled", "#F6F7FB"), ("active", "#FFFFFF")],
        )
        style.configure(
            "Nav.TButton",
            padding=(10, 8),
            background=self.palette["sidebar"],
            foreground=self.palette["text"],
            borderwidth=0,
            relief="flat",
            anchor="w",
            font=(self.ui_font, 10),
        )
        style.map(
            "Nav.TButton",
            background=[("active", "#DFE0E8"), ("pressed", "#D6D8E4")],
        )
        style.configure(
            "NavActive.TButton",
            padding=(10, 8),
            background=self.palette["surface"],
            foreground=self.palette["text"],
            borderwidth=0,
            relief="flat",
            anchor="w",
            font=(self.ui_font, 10, "bold"),
        )
        style.map(
            "NavActive.TButton",
            background=[("active", self.palette["surface"]), ("pressed", "#EFF0F5")],
        )
        style.configure(
            "NavIcon.TButton",
            padding=(6, 6),
            background=self.palette["sidebar"],
            foreground=self.palette["text"],
            borderwidth=0,
            relief="flat",
            anchor="center",
            font=(self.ui_font, 12),
        )
        style.map(
            "NavIcon.TButton",
            background=[("active", "#DFE0E8"), ("pressed", "#D6D8E4")],
        )
        style.configure(
            "NavIconActive.TButton",
            padding=(6, 6),
            background=self.palette["surface"],
            foreground=self.palette["text"],
            borderwidth=0,
            relief="flat",
            anchor="center",
            font=(self.ui_font, 12),
        )
        style.map(
            "NavIconActive.TButton",
            background=[("active", self.palette["surface"]), ("pressed", "#EFF0F5")],
        )
        style.configure(
            "NavLabel.TLabel",
            background=self.palette["sidebar"],
            foreground=self.palette["text"],
            font=(self.ui_font, 10),
        )
        style.configure(
            "QuickLabel.TLabel",
            background=self.palette["sidebar"],
            foreground=self.palette["text"],
            font=(self.ui_font, 9),
        )
        style.configure(
            "Rail.TButton",
            padding=(8, 7),
            background=self.palette["rail"],
            foreground=self.palette["muted"],
            borderwidth=0,
            relief="flat",
            anchor="center",
            font=(self.ui_font, 10, "bold"),
        )
        style.map(
            "Rail.TButton",
            background=[("active", "#D7D7E1"), ("pressed", "#CDCFDA")],
            foreground=[("active", self.palette["text"])],
        )
        style.configure(
            "RailActive.TButton",
            padding=(8, 7),
            background=self.palette["surface"],
            foreground=self.palette["text"],
            borderwidth=0,
            relief="flat",
            anchor="center",
            font=(self.ui_font, 10, "bold"),
        )

        style.configure(
            "TEntry",
            fieldbackground=self.palette["surface"],
            foreground=self.palette["text"],
            relief="solid",
            padding=(6, 5),
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.palette["surface"],
            background=self.palette["surface"],
            foreground=self.palette["text"],
            relief="solid",
            arrowsize=14,
            padding=(6, 5),
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.palette["surface"])],
            foreground=[("readonly", self.palette["text"])],
            selectbackground=[("readonly", self.palette["surface"])],
            selectforeground=[("readonly", self.palette["text"])],
        )
        style.configure(
            "TCheckbutton",
            background=self.palette["surface"],
            foreground=self.palette["text"],
            font=(self.ui_font, 10),
        )
        style.configure(
            "Treeview",
            background=self.palette["surface"],
            fieldbackground=self.palette["surface"],
            foreground=self.palette["text"],
            rowheight=30,
            borderwidth=1,
            relief="solid",
            font=(self.ui_font, 10),
        )
        style.configure(
            "Treeview.Heading",
            background=self.palette["surface_alt"],
            foreground=self.palette["text"],
            relief="flat",
            font=(self.ui_font, 10, "bold"),
            borderwidth=0,
            padding=(6, 6),
        )
        style.map(
            "Treeview",
            background=[("selected", self.palette["selection"])],
            foreground=[("selected", self.palette["text"])],
        )
        style.configure(
            "Vertical.TScrollbar",
            background="#D7D9E2",
            troughcolor=self.palette["surface"],
            arrowsize=14,
        )

    def _build_layout(self) -> None:
        root_shell = ttk.Frame(self.root, style="Root.TFrame", padding=(26, 22, 26, 22))
        root_shell.grid(row=0, column=0, sticky="nsew")
        root_shell.rowconfigure(0, weight=1)
        root_shell.columnconfigure(0, weight=1)

        app_shell = ttk.Frame(root_shell, style="Shell.TFrame")
        app_shell.grid(row=0, column=0, sticky="nsew")
        app_shell.rowconfigure(0, weight=1)
        app_shell.columnconfigure(0, minsize=68)
        app_shell.columnconfigure(1, minsize=240)
        app_shell.columnconfigure(2, weight=1)

        rail = ttk.Frame(app_shell, style="Rail.TFrame", padding=(10, 14, 10, 14))
        rail.grid(row=0, column=0, sticky="ns")
        rail.rowconfigure(10, weight=1)

        sidebar = ttk.Frame(app_shell, style="Sidebar.TFrame", padding=(14, 14, 14, 14))
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.rowconfigure(4, weight=1)
        sidebar.columnconfigure(0, weight=1)

        main = ttk.Frame(app_shell, style="Main.TFrame", padding=(18, 16, 18, 14))
        main.grid(row=0, column=2, sticky="nsew")
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        self._build_left_rail(rail)
        self._build_sidebar(sidebar)

        header = ttk.Frame(main, style="HeaderCard.TFrame", padding=(14, 12, 14, 12))
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, textvariable=self.page_title_var, style="AppTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(header, textvariable=self.page_subtitle_var, style="AppSubtitle.TLabel").grid(
            row=1, column=0, sticky="w", pady=(2, 0)
        )

        header_right = ttk.Frame(header, style="HeaderCard.TFrame")
        header_right.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(header_right, text="Last meeting", style="AppSubtitle.TLabel").grid(
            row=0,
            column=0,
            sticky="e",
        )
        ttk.Label(header_right, textvariable=self.last_meeting_var, style="Badge.TLabel").grid(
            row=1,
            column=0,
            sticky="e",
            pady=(4, 0),
        )

        content_host = ttk.Frame(main, style="Main.TFrame")
        content_host.grid(row=1, column=0, sticky="nsew")
        content_host.rowconfigure(0, weight=1)
        content_host.columnconfigure(0, weight=1)

        self.page_frames = {
            "transcribe": ttk.Frame(content_host, style="App.TFrame", padding=(2, 8, 2, 2)),
            "summarize": ttk.Frame(content_host, style="App.TFrame", padding=(2, 8, 2, 2)),
            "search": ttk.Frame(content_host, style="App.TFrame", padding=(2, 8, 2, 2)),
            "export": ttk.Frame(content_host, style="App.TFrame", padding=(2, 8, 2, 2)),
            "health": ttk.Frame(content_host, style="App.TFrame", padding=(2, 8, 2, 2)),
        }
        for frame in self.page_frames.values():
            frame.grid(row=0, column=0, sticky="nsew")

        self._build_transcribe_tab(self.page_frames["transcribe"])
        self._build_summarize_tab(self.page_frames["summarize"])
        self._build_search_tab(self.page_frames["search"])
        self._build_export_tab(self.page_frames["export"])
        self._build_doctor_tab(self.page_frames["health"])

        status_bar = ttk.Frame(main, style="Status.TFrame", padding=(10, 6))
        status_bar.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        status_bar.columnconfigure(0, weight=1)
        ttk.Label(status_bar, textvariable=self.status_var, style="StatusText.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(status_bar, text="offline mode", style="StatusText.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
        )

        self._show_page("transcribe")

    def _build_left_rail(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="BRAD", style="RailLabel.TLabel").grid(
            row=0, column=0, sticky="ew", pady=(2, 14)
        )

        for row_index, key in enumerate(self.page_meta, start=1):
            button = ttk.Button(
                parent,
                text=self.page_icons.get(key, "\u2022"),
                width=3,
                style="Rail.TButton",
                command=lambda page_key=key: self._show_page(page_key),
            )
            button.grid(row=row_index, column=0, sticky="ew", pady=4)
            self.rail_buttons[key] = button

        ttk.Label(parent, text="...", style="RailLabel.TLabel").grid(
            row=11, column=0, sticky="s", pady=(10, 2)
        )

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Channels", style="SidebarTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 4)
        )
        ttk.Label(
            parent,
            text="Switch between Brad workflows",
            style="SidebarHint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        nav_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        nav_frame.grid(row=2, column=0, sticky="ew")
        nav_frame.columnconfigure(0, weight=1)
        nav_frame.columnconfigure(1, weight=1)

        for row_index, (key, (title, _)) in enumerate(self.page_meta.items()):
            icon_button = ttk.Button(
                nav_frame,
                text=self.page_icons.get(key, "\u2022"),
                width=3,
                style="NavIcon.TButton",
                command=lambda page_key=key: self._show_page(page_key),
            )
            icon_button.grid(row=row_index, column=0, sticky="w", pady=2, padx=(0, 6))
            self.nav_buttons[key] = icon_button

            label = ttk.Label(nav_frame, text=title, style="NavLabel.TLabel")
            label.grid(row=row_index, column=1, sticky="w", pady=2)
            label.bind("<Button-1>", lambda _event, page_key=key: self._show_page(page_key))

        quick_actions = ttk.LabelFrame(parent, text="Quick Actions", style="SidebarCard.TLabelframe")
        quick_actions.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        quick_actions.columnconfigure(1, weight=1)

        summary_btn = ttk.Button(
            quick_actions,
            text=self.action_icons["summary"],
            width=3,
            style="NavIcon.TButton",
            command=self._use_last_for_summary,
        )
        summary_btn.grid(row=0, column=0, sticky="w", pady=(0, 6), padx=(0, 6))
        summary_lbl = ttk.Label(quick_actions, text="Use Last For Summary", style="QuickLabel.TLabel")
        summary_lbl.grid(row=0, column=1, sticky="w", pady=(0, 6))
        summary_lbl.bind("<Button-1>", lambda _event: self._use_last_for_summary())

        export_btn = ttk.Button(
            quick_actions,
            text=self.action_icons["export"],
            width=3,
            style="NavIcon.TButton",
            command=self._use_last_for_export,
        )
        export_btn.grid(row=1, column=0, sticky="w", pady=(0, 6), padx=(0, 6))
        export_lbl = ttk.Label(quick_actions, text="Use Last For Export", style="QuickLabel.TLabel")
        export_lbl.grid(row=1, column=1, sticky="w", pady=(0, 6))
        export_lbl.bind("<Button-1>", lambda _event: self._use_last_for_export())

        doctor_btn = ttk.Button(
            quick_actions,
            text=self.action_icons["doctor"],
            width=3,
            style="NavIcon.TButton",
            command=self._on_doctor,
        )
        doctor_btn.grid(row=2, column=0, sticky="w", padx=(0, 6))
        doctor_lbl = ttk.Label(quick_actions, text="Run Doctor", style="QuickLabel.TLabel")
        doctor_lbl.grid(row=2, column=1, sticky="w")
        doctor_lbl.bind("<Button-1>", lambda _event: self._on_doctor())

        ttk.Label(parent, text="Last meeting ID", style="SidebarHint.TLabel").grid(
            row=5, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Label(parent, textvariable=self.last_meeting_var, style="Badge.TLabel").grid(
            row=6, column=0, sticky="w", pady=(4, 0)
        )

    def _show_page(self, page_key: str) -> None:
        frame = self.page_frames.get(page_key)
        if frame is None:
            return

        frame.tkraise()
        title, subtitle = self.page_meta.get(page_key, ("Brad", ""))
        self.page_title_var.set(title)
        self.page_subtitle_var.set(subtitle)

        for key, button in self.nav_buttons.items():
            button.configure(
                style="NavIconActive.TButton" if key == page_key else "NavIcon.TButton"
            )
        for key, button in self.rail_buttons.items():
            button.configure(style="RailActive.TButton" if key == page_key else "Rail.TButton")

    def _build_transcribe_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        source_frame = ttk.LabelFrame(parent, text="Audio Input", style="Card.TLabelframe")
        source_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        source_frame.columnconfigure(1, weight=1)

        ttk.Label(source_frame, text="Audio file").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(source_frame, textvariable=self.audio_path_var).grid(
            row=0, column=1, sticky="ew", padx=(0, 6)
        )
        ttk.Button(source_frame, text="Browse...", command=self._pick_audio_file).grid(
            row=0, column=2, sticky="ew"
        )

        options_frame = ttk.LabelFrame(parent, text="Transcription Settings", style="Card.TLabelframe")
        options_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        options_frame.columnconfigure(6, weight=1)

        ttk.Label(options_frame, text="Model").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            options_frame,
            textvariable=self.model_var,
            values=("small", "medium", "large"),
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="w", padx=(4, 12))

        ttk.Label(options_frame, text="Language").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            options_frame,
            textvariable=self.language_var,
            values=("auto", "cs", "en"),
            state="readonly",
            width=12,
        ).grid(row=0, column=3, sticky="w", padx=(4, 12))

        ttk.Checkbutton(options_frame, text="Use Silero VAD", variable=self.use_vad_var).grid(
            row=0, column=4, sticky="w"
        )

        transcribe_button = ttk.Button(
            options_frame,
            text="Start Transcription",
            command=self._on_transcribe,
            style="Primary.TButton",
        )
        transcribe_button.grid(row=0, column=5, sticky="e", padx=(12, 0))
        self._action_buttons.append(transcribe_button)

        result_pane = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        result_pane.grid(row=2, column=0, sticky="nsew")

        transcript_frame = ttk.LabelFrame(result_pane, text="Transcript Preview", style="Card.TLabelframe")
        transcript_frame.columnconfigure(0, weight=1)
        transcript_frame.rowconfigure(0, weight=1)
        self.transcript_preview = ScrolledText(
            transcript_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            undo=False,
            background="#F8FAFD",
            foreground=self.palette["text"],
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            insertbackground=self.palette["text"],
        )
        self.transcript_preview.grid(row=0, column=0, sticky="nsew")
        self._set_text(self.transcript_preview, "(transcript preview will appear here)")
        self.transcript_preview.configure(state=tk.DISABLED)

        exports_frame = ttk.LabelFrame(result_pane, text="Generated Exports", style="Card.TLabelframe")
        exports_frame.columnconfigure(0, weight=1)
        exports_frame.rowconfigure(0, weight=1)
        self.export_paths_box = ScrolledText(
            exports_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            undo=False,
            width=42,
            background="#F8FAFD",
            foreground=self.palette["text"],
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            insertbackground=self.palette["text"],
        )
        self.export_paths_box.grid(row=0, column=0, sticky="nsew")
        self._set_text(self.export_paths_box, "(export paths will appear here)")
        self.export_paths_box.configure(state=tk.DISABLED)

        result_pane.add(transcript_frame, weight=4)
        result_pane.add(exports_frame, weight=3)

    def _build_summarize_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(parent, text="Summary Input", style="Card.TLabelframe")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(6, weight=1)

        ttk.Label(controls, text="Target").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.summarize_target_var).grid(
            row=0, column=1, sticky="ew", padx=(4, 6)
        )
        ttk.Button(controls, text="Use Last Meeting", command=self._use_last_for_summary).grid(
            row=0, column=2, sticky="ew", padx=(0, 6)
        )
        ttk.Button(controls, text="Pick Transcript...", command=self._pick_transcript_file).grid(
            row=0, column=3, sticky="ew", padx=(0, 6)
        )

        ttk.Label(controls, text="Template").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.template_var,
            values=("general", "sales", "engineering"),
            state="readonly",
            width=16,
        ).grid(row=1, column=1, sticky="w", padx=(4, 6), pady=(8, 0))

        ttk.Label(controls, text="LLM model (optional)").grid(
            row=1,
            column=2,
            sticky="e",
            pady=(8, 0),
        )
        ttk.Entry(controls, textvariable=self.llm_model_var).grid(
            row=1, column=3, columnspan=2, sticky="ew", padx=(4, 6), pady=(8, 0)
        )
        ttk.Button(controls, text="Browse...", command=self._pick_llm_model).grid(
            row=1, column=5, sticky="ew", padx=(0, 6), pady=(8, 0)
        )

        summarize_button = ttk.Button(
            controls,
            text="Generate Summary",
            command=self._on_summarize,
            style="Primary.TButton",
        )
        summarize_button.grid(row=1, column=6, sticky="e", pady=(8, 0))
        self._action_buttons.append(summarize_button)

        summary_frame = ttk.LabelFrame(parent, text="Summary Output", style="Card.TLabelframe")
        summary_frame.grid(row=1, column=0, sticky="nsew")
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        self.summary_output = ScrolledText(
            summary_frame,
            wrap=tk.WORD,
            font=(self.ui_font, 10),
            background="#F8FAFD",
            foreground=self.palette["text"],
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            insertbackground=self.palette["text"],
        )
        self.summary_output.grid(row=0, column=0, sticky="nsew")
        self._set_text(self.summary_output, "(summary output will appear here)")
        self.summary_output.configure(state=tk.DISABLED)

    def _build_search_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(parent, text="Search Query", style="Card.TLabelframe")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)

        ttk.Label(controls, text="Query").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.search_query_var).grid(
            row=0, column=1, sticky="ew", padx=(4, 8)
        )

        ttk.Label(controls, text="Meeting ID (optional)").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.search_meeting_var, width=14).grid(
            row=0, column=3, sticky="w", padx=(4, 8)
        )

        search_button = ttk.Button(
            controls,
            text="Search",
            command=self._on_search,
            style="Primary.TButton",
        )
        search_button.grid(row=0, column=4, sticky="e")
        self._action_buttons.append(search_button)

        table_frame = ttk.LabelFrame(parent, text="Search Results", style="Card.TLabelframe")
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.search_tree = ttk.Treeview(
            table_frame,
            columns=("meeting", "segment", "start", "end", "snippet"),
            show="headings",
        )
        self.search_tree.heading("meeting", text="Meeting")
        self.search_tree.heading("segment", text="Segment")
        self.search_tree.heading("start", text="Start")
        self.search_tree.heading("end", text="End")
        self.search_tree.heading("snippet", text="Snippet")
        self.search_tree.column("meeting", width=80, anchor="center")
        self.search_tree.column("segment", width=80, anchor="center")
        self.search_tree.column("start", width=90, anchor="e")
        self.search_tree.column("end", width=90, anchor="e")
        self.search_tree.column("snippet", width=700, anchor="w")

        scroll = ttk.Scrollbar(
            table_frame,
            orient=tk.VERTICAL,
            command=self.search_tree.yview,
            style="Vertical.TScrollbar",
        )
        self.search_tree.configure(yscrollcommand=scroll.set)
        self.search_tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def _build_export_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        controls = ttk.LabelFrame(parent, text="Export", style="Card.TLabelframe")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(5, weight=1)

        ttk.Label(controls, text="Meeting ID").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.export_meeting_var, width=12).grid(
            row=0, column=1, sticky="w", padx=(4, 6)
        )
        ttk.Button(controls, text="Use Last Meeting", command=self._use_last_for_export).grid(
            row=0, column=2, sticky="w", padx=(0, 8)
        )

        ttk.Label(controls, text="Format").grid(row=0, column=3, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.export_format_var,
            values=("md", "srt", "json"),
            state="readonly",
            width=10,
        ).grid(row=0, column=4, sticky="w", padx=(4, 8))

        export_button = ttk.Button(
            controls,
            text="Export",
            command=self._on_export,
            style="Primary.TButton",
        )
        export_button.grid(row=0, column=5, sticky="e")
        self._action_buttons.append(export_button)

        output_frame = ttk.LabelFrame(parent, text="Output Path", style="Card.TLabelframe")
        output_frame.grid(row=1, column=0, sticky="ew")
        output_frame.columnconfigure(0, weight=1)
        ttk.Entry(output_frame, textvariable=self.export_path_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(output_frame, text="Open File", command=self._open_export_file).grid(
            row=0, column=1, padx=(0, 6)
        )
        ttk.Button(output_frame, text="Open Folder", command=self._open_export_folder).grid(
            row=0, column=2
        )

    def _build_doctor_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.Frame(parent, style="App.TFrame")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(1, weight=1)
        doctor_button = ttk.Button(
            controls,
            text="Run Doctor",
            command=self._on_doctor,
            style="Primary.TButton",
        )
        doctor_button.grid(row=0, column=0, sticky="w")
        self._action_buttons.append(doctor_button)

        table_frame = ttk.LabelFrame(parent, text="Runtime Checks", style="Card.TLabelframe")
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.doctor_tree = ttk.Treeview(
            table_frame,
            columns=("check", "status", "detail"),
            show="headings",
        )
        self.doctor_tree.heading("check", text="Check")
        self.doctor_tree.heading("status", text="Status")
        self.doctor_tree.heading("detail", text="Detail")
        self.doctor_tree.column("check", width=240, anchor="w")
        self.doctor_tree.column("status", width=90, anchor="center")
        self.doctor_tree.column("detail", width=760, anchor="w")
        self.doctor_tree.tag_configure("ok", foreground="#166534")
        self.doctor_tree.tag_configure("warn", foreground="#B45309")
        self.doctor_tree.tag_configure("fail", foreground="#B91C1C")

        scroll = ttk.Scrollbar(
            table_frame,
            orient=tk.VERTICAL,
            command=self.doctor_tree.yview,
            style="Vertical.TScrollbar",
        )
        self.doctor_tree.configure(yscrollcommand=scroll.set)
        self.doctor_tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def _pick_audio_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.audio_path_var.set(selected)

    def _pick_transcript_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select transcript file",
            filetypes=[
                ("Transcript", "*.txt *.md *.json *.srt"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.summarize_target_var.set(selected)

    def _pick_llm_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select GGUF model",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
        )
        if selected:
            self.llm_model_var.set(selected)

    def _use_last_for_summary(self) -> None:
        if self.last_meeting_id is not None:
            self.summarize_target_var.set(str(self.last_meeting_id))

    def _use_last_for_export(self) -> None:
        if self.last_meeting_id is not None:
            value = str(self.last_meeting_id)
            self.export_meeting_var.set(value)
            self.search_meeting_var.set(value)

    def _on_transcribe(self) -> None:
        raw_path = self.audio_path_var.get().strip()
        if not raw_path:
            messagebox.showerror("Brad", "Select an audio file first.")
            return
        audio_path = Path(raw_path).expanduser()
        if not audio_path.exists():
            messagebox.showerror("Brad", f"Audio file not found:\n{audio_path}")
            return

        model = self.model_var.get().strip()
        language = self.language_var.get().strip()
        use_vad = bool(self.use_vad_var.get())

        def task() -> TranscriptionOutcome:
            return self.service.transcribe_file(
                audio_path,
                model_name=model,
                language=language,
                use_vad=use_vad,
            )

        def done(outcome: TranscriptionOutcome) -> None:
            self.last_meeting_id = outcome.meeting_id
            self.last_meeting_var.set(str(outcome.meeting_id))
            meeting_str = str(outcome.meeting_id)
            self.summarize_target_var.set(meeting_str)
            self.search_meeting_var.set(meeting_str)
            self.export_meeting_var.set(meeting_str)

            segments = self.service.db.get_segments(outcome.meeting_id)
            preview = "\n".join(
                f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}"
                for segment in segments[:120]
            )
            exports = "\n".join(f"{fmt}: {path}" for fmt, path in outcome.export_paths.items())
            self._set_text(self.transcript_preview, preview or "(no transcript segments)")
            self._set_text(self.export_paths_box, exports or "(no exports)")
            self.status_var.set(
                "Meeting "
                f"{outcome.meeting_id} ready "
                f"({outcome.segment_count} segments, {outcome.language})."
            )

        self._run_async("Transcribing audio...", task, done)

    def _on_summarize(self) -> None:
        target = self.summarize_target_var.get().strip()
        if not target:
            messagebox.showerror("Brad", "Provide meeting ID or transcript file path.")
            return

        template = self.template_var.get().strip()
        llm_raw = self.llm_model_var.get().strip()
        llm_model = Path(llm_raw).expanduser() if llm_raw else None

        def task() -> SummaryOutcome:
            return self.service.summarize_target(
                target,
                template_name=template,
                llm_model=llm_model,
            )

        def done(outcome: SummaryOutcome) -> None:
            if outcome.meeting_id is not None:
                self.last_meeting_id = outcome.meeting_id
                self.last_meeting_var.set(str(outcome.meeting_id))
            self._set_text(self.summary_output, outcome.summary.text)
            self.status_var.set(f"Summary generated via {outcome.summary.method}.")

        self._run_async("Generating summary...", task, done)

    def _on_search(self) -> None:
        query = self.search_query_var.get().strip()
        if not query:
            messagebox.showerror("Brad", "Provide search query.")
            return

        meeting_text = self.search_meeting_var.get().strip()
        if meeting_text and not meeting_text.isdigit():
            messagebox.showerror("Brad", "Meeting ID must be numeric.")
            return
        meeting_id = int(meeting_text) if meeting_text else None

        def task() -> list[SearchHit]:
            return self.service.search(query, meeting_id=meeting_id, limit=100)

        def done(hits: list[SearchHit]) -> None:
            for item in self.search_tree.get_children():
                self.search_tree.delete(item)
            for hit in hits:
                self.search_tree.insert(
                    "",
                    tk.END,
                    values=(
                        hit.meeting_id,
                        hit.segment_id,
                        f"{hit.start:.2f}",
                        f"{hit.end:.2f}",
                        hit.snippet.replace("\n", " "),
                    ),
                )
            self.status_var.set(f"Search done. {len(hits)} result(s).")

        self._run_async("Searching transcripts...", task, done)

    def _on_export(self) -> None:
        meeting_raw = self.export_meeting_var.get().strip()
        if not meeting_raw.isdigit():
            messagebox.showerror("Brad", "Meeting ID must be numeric.")
            return
        meeting_id = int(meeting_raw)
        export_format = self.export_format_var.get().strip()

        def task() -> Path:
            return self.service.export_meeting(meeting_id, export_format)

        def done(path: Path) -> None:
            self.export_path_var.set(str(path))
            self.status_var.set(f"Export created: {path}")

        self._run_async(f"Exporting meeting {meeting_id}...", task, done)

    def _on_doctor(self) -> None:
        def task() -> list[DoctorCheck]:
            return run_doctor(self.service.settings)

        def done(checks: list[DoctorCheck]) -> None:
            for item in self.doctor_tree.get_children():
                self.doctor_tree.delete(item)
            for check in checks:
                self.doctor_tree.insert(
                    "",
                    tk.END,
                    values=(check.name, check.status.upper(), check.detail),
                    tags=(check.status,),
                )
            failing = sum(1 for check in checks if check.status == "fail")
            warning = sum(1 for check in checks if check.status == "warn")
            self.status_var.set(
                f"Doctor completed ({failing} fail, {warning} warn, {len(checks)} checks)."
            )

        self._run_async("Running environment checks...", task, done)

    def _open_export_file(self) -> None:
        path_text = self.export_path_var.get().strip()
        if not path_text:
            messagebox.showinfo("Brad", "No exported file path yet.")
            return
        self._open_path(Path(path_text))

    def _open_export_folder(self) -> None:
        path_text = self.export_path_var.get().strip()
        if not path_text:
            messagebox.showinfo("Brad", "No exported file path yet.")
            return
        self._open_path(Path(path_text).parent)

    def _open_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror("Brad", f"Path does not exist:\n{path}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # pragma: no cover - platform specific
            messagebox.showerror("Brad", f"Failed to open path:\n{exc}")

    def _set_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for button in self._action_buttons:
            button.configure(state=state)
        self.status_var.set(status)

    def _run_async(
        self,
        busy_status: str,
        task_fn: Callable[[], object],
        on_success: Callable[[object], None],
    ) -> None:
        if self._busy:
            messagebox.showinfo("Brad", "Another task is already running. Please wait.")
            return

        self._set_busy(True, busy_status)
        future = self.executor.submit(task_fn)
        self._poll_future(future, on_success)

    def _poll_future(self, future: Future[object], on_success: Callable[[object], None]) -> None:
        if not future.done():
            self.root.after(120, self._poll_future, future, on_success)
            return

        try:
            result = future.result()
        except Exception as exc:
            self._set_busy(False, f"Error: {exc}")
            messagebox.showerror("Brad", str(exc))
            return

        try:
            on_success(result)
        except Exception as exc:  # pragma: no cover - defensive UI path
            self._set_busy(False, f"UI update error: {exc}")
            messagebox.showerror("Brad", f"UI update failed: {exc}")
            return

        if not self._busy:
            return
        self._set_busy(False, self.status_var.get())

    def _on_close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def launch_desktop_app() -> None:
    root = tk.Tk()
    app = BradDesktopApp(root)
    app._on_doctor()
    root.mainloop()
