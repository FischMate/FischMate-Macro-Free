from __future__ import annotations

import ctypes
import sys
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk

from app import __version__
from app.branding import APP_NAME, WINDOWS_APP_ID
from app.capture.windows import (
    WindowInfo,
    activate_window,
    enumerate_roblox_windows,
    is_foreground_window,
)
from app.config.profiles import ProfileError, ProfileRepository
from app.gui.status_overlay import StatusOverlay
from app.gui.fonts import bold, medium, monospace, regular, semibold
from app.hotkeys import GlobalHotkeyWatcher
from app.live.engine import LiveDetectionEngine, LiveStatus
from app.windows_taskbar import set_taskbar_identity


# Keep these visible while FischMate is being developed. Set this to False for
# the public release; launch_debug.bat will continue to expose them.
SHOW_DEVELOPER_CONTROLS_DURING_BUILD = False

DISCORD_INVITE_URL = "https://discord.com/invite/rFwGCcECce"
FISCHMATE_UPGRADE_URL = "https://fischmate.com"


COLORS = {
    "window": "#ffffff",
    "sidebar": "#f8fafc",
    "line": "#dce3ea",
    "text": "#172033",
    "muted": "#667085",
    "blue": "#2084ef",
    "blue_dark": "#126bd1",
    "blue_soft": "#edf7ff",
    "blue_border": "#b7d9f5",
    "green": "#24933c",
    "green_soft": "#eefbef",
    "green_border": "#94d39a",
    "red": "#dc4247",
    "red_soft": "#fff5f5",
    "card": "#ffffff",
}


def ease_out_cubic(progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return 1.0 - (1.0 - progress) ** 3


def mix_hex(start: str, end: str, progress: float) -> str:
    progress = min(1.0, max(0.0, progress))
    start_rgb = tuple(int(start[index:index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index:index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(round(a + (b - a) * progress) for a, b in zip(start_rgb, end_rgb))
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


class ActionCanvasButton(tk.Canvas):
    """Rounded action button with a code-drawn circular action badge."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        text: str,
        command,
        kind: str,
        badge_image: tk.PhotoImage | None = None,
        width: int = 265,
        height: int = 50,
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=parent.cget("bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.button_text = text
        self.command = command
        self.kind = kind
        self.badge_image = badge_image
        self.button_width = width
        self.button_height = height
        self.enabled = True
        self.hovered = False
        self.pressed = False
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self._draw()

    def _rounded_rectangle(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        radius: int,
        **kwargs,
    ) -> int:
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, splinesteps=24, **kwargs)

    def _draw(self) -> None:
        self.delete("all")
        inset = 3 if self.pressed else 1
        content_offset = 1 if self.pressed else 0
        if self.kind == "primary":
            fill = COLORS["blue_dark"] if self.pressed else "#2d91f5" if self.hovered else COLORS["blue"]
            if not self.enabled:
                fill = "#a7caeb"
            self._rounded_rectangle(
                inset, inset, self.button_width - inset, self.button_height - inset, 10,
                fill=fill,
                outline=fill,
            )
            text_fill = "white"
            badge_x = 50
            if self.badge_image is not None:
                self.create_image(badge_x, self.button_height / 2 + content_offset, image=self.badge_image)
            else:
                symbol_fill = COLORS["blue"] if self.enabled else "#86add1"
                self.create_oval(35, 12, 61, 38, fill="white", outline="")
                self.create_polygon(
                    badge_x - 3, 18,
                    badge_x - 3, 32,
                    badge_x + 8, 25,
                    fill=symbol_fill,
                    outline=symbol_fill,
                )
        else:
            fill = "#ffe3e3" if self.pressed else "#fff0f0" if self.hovered else COLORS["red_soft"]
            if not self.enabled:
                fill = "#faf7f7"
            self._rounded_rectangle(
                inset, inset, self.button_width - inset, self.button_height - inset, 10,
                fill=fill,
                outline=COLORS["red"] if self.enabled else "#d9b5b5",
                width=1,
            )
            text_fill = COLORS["red"] if self.enabled else "#b98b8b"
            badge_x = 50
            if self.badge_image is not None:
                self.create_image(badge_x, self.button_height / 2 + content_offset, image=self.badge_image)
            else:
                self.create_oval(38, 13, 62, 37, fill=COLORS["red"], outline="")
                self.create_rectangle(46, 21, 54, 29, fill="white", outline="white")
        self.create_text(
            72,
            self.button_height / 2 + content_offset,
            text=self.button_text,
            anchor="w",
            fill=text_fill,
            font=semibold(11),
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw()

    def _enter(self, _event) -> None:
        if self.enabled:
            self.hovered = True
            self._draw()

    def _leave(self, _event) -> None:
        self.hovered = False
        self.pressed = False
        self._draw()

    def _press(self, _event) -> None:
        if self.enabled:
            self.pressed = True
            self._draw()

    def _release(self, event) -> None:
        was_pressed = self.pressed
        self.pressed = False
        self._draw()
        inside = 0 <= event.x < self.button_width and 0 <= event.y < self.button_height
        if self.enabled and was_pressed and inside:
            self.after(25, self.command)


class SearchableRodDropdown(tk.Frame):
    """Searchable rod picker matching FischMate's setup-card visual language."""

    TRIGGER_HEIGHT = 42
    ROW_HEIGHT = 34
    MAX_VISIBLE_ROWS = 7

    def __init__(
        self,
        parent: tk.Widget,
        *,
        variable: tk.StringVar,
        values: list[str],
        command,
        search_icon: tk.PhotoImage,
        scrollbar_thumb: tk.PhotoImage,
        scroll_up_arrow: tk.PhotoImage,
        scroll_down_arrow: tk.PhotoImage,
        trigger_default_strip: tk.PhotoImage,
        trigger_default_right: tk.PhotoImage,
        trigger_open_strip: tk.PhotoImage,
        trigger_open_right: tk.PhotoImage,
        trigger_arrow_down: tk.PhotoImage,
        trigger_arrow_up: tk.PhotoImage,
        searchable: bool = True,
        sort_values: bool = True,
        reserve_icon_space: bool = True,
        placeholder: str = "Choose a rod",
        empty_message: str = "No rods found",
        max_visible_rows: int | None = None,
        item_badges: dict[str, tk.PhotoImage] | None = None,
        item_states: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            parent,
            height=self.TRIGGER_HEIGHT,
            bg=parent.cget("bg"),
            highlightthickness=0,
            bd=0,
        )
        self.pack_propagate(False)
        self.grid_propagate(False)
        self.variable = variable
        self.sort_values = sort_values
        self.values = self._ordered_values(values)
        self.command = command
        self.searchable = searchable
        self.reserve_icon_space = reserve_icon_space
        self.placeholder = placeholder
        self.empty_message = empty_message
        self.max_visible_rows = max_visible_rows or self.MAX_VISIBLE_ROWS
        self.item_badges = item_badges or {}
        self.item_states = item_states or {}
        self.search_icon = search_icon
        self.scrollbar_thumb = scrollbar_thumb
        self.scroll_up_arrow = scroll_up_arrow
        self.scroll_down_arrow = scroll_down_arrow
        self.trigger_default_strip = trigger_default_strip
        self.trigger_default_right = trigger_default_right
        self.trigger_open_strip = trigger_open_strip
        self.trigger_open_right = trigger_open_right
        self.trigger_arrow_down = trigger_arrow_down
        self.trigger_arrow_up = trigger_arrow_up
        self.enabled = True
        self._popup: tk.Toplevel | None = None
        self._search_var = tk.StringVar()
        self._search_trace = self._search_var.trace_add("write", self._search_changed)
        self._rows: dict[str, tk.Canvas] = {}
        self._filtered_values: list[str] = []
        self._active_index = 0
        self._variable_trace = self.variable.trace_add("write", self._variable_changed)
        self._trigger_pressed = False
        self._popup_animation_job: str | None = None

        self.trigger = tk.Canvas(
            self,
            height=self.TRIGGER_HEIGHT,
            bg=parent.cget("bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.trigger.pack(fill="both", expand=True)
        self.trigger.bind("<Configure>", lambda _event: self._draw_trigger())
        self.trigger.bind("<ButtonPress-1>", self._trigger_press)
        self.trigger.bind("<ButtonRelease-1>", self._trigger_release)
        self.trigger.bind("<Return>", self._toggle_popup)
        self.trigger.bind("<space>", self._toggle_popup)
        self.trigger.configure(takefocus=True)

    def _ordered_values(self, values: list[str]) -> list[str]:
        return sorted(values, key=str.casefold) if self.sort_values else list(values)

    def set_values(self, values: list[str]) -> None:
        self.values = self._ordered_values(values)
        if self._popup is not None and self._popup.winfo_exists():
            self._render_rows()

    def current(self, index: int | None = None) -> int:
        if index is not None:
            if 0 <= index < len(self.values):
                self.variable.set(self.values[index])
            return index
        try:
            return self.values.index(self.variable.get())
        except ValueError:
            return -1

    @staticmethod
    def filter_labels(values: list[str], query: str) -> list[str]:
        terms = query.casefold().split()
        return sorted(
            (
                label
                for label in values
                if all(term in label.casefold() for term in terms)
            ),
            key=str.casefold,
        )

    def _variable_changed(self, *_args) -> None:
        if self.winfo_exists():
            self.after_idle(self._draw_trigger)

    def _draw_trigger(self) -> None:
        if not self.trigger.winfo_exists():
            return
        self.trigger.delete("all")
        width = max(180, self.trigger.winfo_width())
        height = max(20, self.trigger.winfo_height())
        opened = self._popup is not None and self._popup.winfo_exists()
        strip = self.trigger_open_strip if opened else self.trigger_default_strip
        right_cap = self.trigger_open_right if opened else self.trigger_default_right
        self.trigger.create_image(0, height / 2, image=strip, anchor="w")
        self.trigger.create_image(width, height / 2, image=right_cap, anchor="e")
        # Keep this space reserved for the per-rod artwork pass.
        content_offset = 1 if self._trigger_pressed else 0
        badge = self.item_badges.get(self.variable.get())
        text_x = 48 if self.reserve_icon_space else 18
        if badge is not None:
            self.trigger.create_image(25, height / 2 + content_offset, image=badge)
            text_x = 48
        self.trigger.create_text(
            text_x,
            height / 2 + content_offset,
            text=self.variable.get() or self.placeholder,
            anchor="w",
            fill=COLORS["text"] if self.enabled else "#98a2b3",
            font=semibold(11),
        )
        arrow = self.trigger_arrow_up if opened else self.trigger_arrow_down
        self.trigger.create_image(width - 27, height / 2 + content_offset, image=arrow)

    def _trigger_press(self, _event=None) -> None:
        if self.enabled:
            self._trigger_pressed = True
            self._draw_trigger()

    def _trigger_release(self, event) -> None:
        was_pressed = self._trigger_pressed
        self._trigger_pressed = False
        self._draw_trigger()
        inside = 0 <= event.x < self.trigger.winfo_width() and 0 <= event.y < self.trigger.winfo_height()
        if self.enabled and was_pressed and inside:
            self._toggle_popup()

    def _toggle_popup(self, _event=None) -> None:
        if not self.enabled:
            return
        if self._popup is not None and self._popup.winfo_exists():
            self._close_popup()
        else:
            self._open_popup()

    def _open_popup(self) -> None:
        self.update_idletasks()
        width = max(420 if self.searchable else 260, self.winfo_width())
        visible_rows = min(self.max_visible_rows, max(1, len(self.values)))
        header_height = 62 if self.searchable else 6
        popup_height = header_height + visible_rows * self.ROW_HEIGHT
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 3
        screen_height = self.winfo_screenheight()
        opens_above = False
        if y + popup_height > screen_height - 12:
            y = max(12, self.winfo_rooty() - popup_height - 3)
            opens_above = True

        popup = tk.Toplevel(self)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.transient(self.winfo_toplevel())
        popup.configure(bg="#d8e0e9")
        popup.geometry(f"{width}x{popup_height}+{x}+{y}")
        popup.bind("<Escape>", lambda _event: self._close_popup())
        popup.bind("<FocusOut>", self._schedule_focus_check)
        popup.bind("<Button-1>", self._close_if_outside, add="+")
        popup.bind("<Down>", self._move_active)
        popup.bind("<Up>", self._move_active)
        popup.bind("<Return>", self._choose_active)
        self._popup = popup

        surface = tk.Frame(
            popup,
            bg="#ffffff",
            highlightbackground="#d5dde7",
            highlightthickness=1,
            bd=0,
        )
        surface.pack(fill="both", expand=True, padx=(1, 3), pady=(1, 4))

        if self.searchable:
            search_border = tk.Frame(
                surface,
                height=36,
                bg="#ffffff",
                highlightbackground="#d4dce6",
                highlightthickness=1,
                bd=0,
            )
            search_border.pack(fill="x", padx=12, pady=(9, 8))
            search_border.pack_propagate(False)
            search_icon = tk.Canvas(
                search_border,
                width=36,
                height=34,
                bg="#ffffff",
                highlightthickness=0,
                bd=0,
            )
            search_icon.pack(side="left")
            search_icon.create_image(18, 17, image=self.search_icon)
            self._search_var.set("")
            self._search_entry = tk.Entry(
                search_border,
                textvariable=self._search_var,
                bg="#ffffff",
                fg=COLORS["text"],
                insertbackground=COLORS["blue"],
                relief="flat",
                bd=0,
                highlightthickness=0,
                font=regular(10),
            )
            self._search_entry.pack(side="left", fill="both", expand=True, padx=(0, 9))
            self._search_placeholder = tk.Label(
                search_border,
                text="Search rods...",
                bg="#ffffff",
                fg="#8b96a8",
                font=regular(10),
                cursor="xterm",
            )
            self._search_placeholder.place(x=37, rely=0.5, anchor="w")
            self._search_placeholder.bind("<Button-1>", self._focus_search_entry)
            search_border.bind("<Button-1>", self._focus_search_entry, add="+")
            search_icon.bind("<Button-1>", self._focus_search_entry)
            self._search_entry.bind("<FocusIn>", self._search_focus_changed)
            self._search_entry.bind("<FocusOut>", self._search_focus_changed)
            self._search_entry.bind("<Down>", self._move_active)
            self._search_entry.bind("<Up>", self._move_active)
            self._search_entry.bind("<Return>", self._choose_active)
            tk.Frame(surface, height=1, bg="#e8edf3").pack(fill="x")
        list_wrap = tk.Frame(surface, bg="#ffffff")
        list_wrap.pack(fill="both", expand=True)
        self._list_canvas = tk.Canvas(
            list_wrap,
            bg="#ffffff",
            highlightthickness=0,
            bd=0,
        )
        self._scrollbar = tk.Canvas(
            list_wrap,
            width=29,
            bd=0,
            bg="#f5f6f9",
            highlightthickness=0,
            cursor="hand2",
        )
        if len(self.values) > visible_rows:
            self._scrollbar.pack(side="right", fill="y", padx=(2, 3), pady=2)
        self._scrollbar.bind("<Configure>", lambda _event: self._draw_scrollbar())
        self._scrollbar.bind("<Button-1>", self._scrollbar_press)
        self._scrollbar.bind("<B1-Motion>", self._scrollbar_drag)
        self._list_canvas.pack(side="left", fill="both", expand=True)
        self._list_canvas.configure(yscrollcommand=self._update_scrollbar)
        self._list_frame = tk.Frame(self._list_canvas, bg="#ffffff")
        self._list_window = self._list_canvas.create_window(
            (0, 0),
            window=self._list_frame,
            anchor="nw",
        )
        self._list_frame.bind("<Configure>", self._update_scroll_region)
        self._list_canvas.bind("<Configure>", self._resize_list_window)
        self._list_canvas.bind("<MouseWheel>", self._mousewheel)

        self._render_rows()
        self._popup_target_height = popup_height
        self._popup_target_y = y
        self._popup_x = x
        self._popup_width = width
        self._popup_opens_above = opens_above
        self._popup_start_y = y + (6 if opens_above else -6)
        popup.geometry(f"{width}x{popup_height}+{x}+{self._popup_start_y}")
        popup.deiconify()
        popup.lift()
        if self.searchable:
            self.after_idle(self._focus_search_entry)
        else:
            popup.focus_force()
        self.after_idle(self._scroll_to_selected)
        self._draw_trigger()
        self._animate_popup_open(0)

    def _animate_popup_open(self, step: int) -> None:
        popup = self._popup
        if popup is None or not popup.winfo_exists():
            self._popup_animation_job = None
            return
        total_steps = 5
        progress = ease_out_cubic((step + 1) / total_steps)
        y = round(self._popup_start_y + (self._popup_target_y - self._popup_start_y) * progress)
        popup.geometry(f"{self._popup_width}x{self._popup_target_height}+{self._popup_x}+{y}")
        if step + 1 < total_steps:
            self._popup_animation_job = self.after(12, self._animate_popup_open, step + 1)
        else:
            self._popup_animation_job = None

    def _search_changed(self, *_args) -> None:
        self._sync_search_placeholder()
        self._render_rows()
        if self._search_var.get().strip():
            self.after_idle(self._scroll_search_results_to_top)

    def _scroll_search_results_to_top(self) -> None:
        if hasattr(self, "_list_canvas") and self._list_canvas.winfo_exists():
            self._list_canvas.yview_moveto(0.0)

    def _focus_search_entry(self, _event=None) -> str:
        if self._popup is not None and self._popup.winfo_exists():
            self._popup.lift()
            self._search_entry.focus_force()
            self._sync_search_placeholder()
        return "break"

    def _search_focus_changed(self, _event=None) -> None:
        self.after_idle(self._sync_search_placeholder)

    def _sync_search_placeholder(self) -> None:
        if (
            not hasattr(self, "_search_placeholder")
            or not self._search_placeholder.winfo_exists()
        ):
            return
        search_has_focus = (
            hasattr(self, "_search_entry")
            and self._search_entry.winfo_exists()
            and self.focus_get() == self._search_entry
        )
        if self._search_var.get() or search_has_focus:
            self._search_placeholder.place_forget()
        else:
            self._search_placeholder.place(x=37, rely=0.5, anchor="w")

    def _render_rows(self) -> None:
        if not hasattr(self, "_list_frame") or not self._list_frame.winfo_exists():
            return
        for child in self._list_frame.winfo_children():
            child.destroy()
        self._rows.clear()
        self._filtered_values = (
            self.filter_labels(self.values, self._search_var.get())
            if self.searchable
            else list(self.values)
        )
        if not self._filtered_values:
            tk.Label(
                self._list_frame,
                text=self.empty_message,
                bg="#ffffff",
                fg=COLORS["muted"],
                font=regular(11),
                pady=18,
            ).pack(fill="x")
            self._active_index = 0
            return
        try:
            self._active_index = self._filtered_values.index(self.variable.get())
        except ValueError:
            self._active_index = 0
        for label in self._filtered_values:
            row = tk.Canvas(
                self._list_frame,
                height=self.ROW_HEIGHT,
                bg="#ffffff",
                highlightthickness=0,
                bd=0,
                cursor="hand2",
            )
            row.pack(fill="x")
            row._rod_hovered = False  # type: ignore[attr-defined]
            row.bind("<Configure>", lambda _event, item=label: self._draw_row(item))
            row.bind("<Enter>", lambda _event, item=label: self._set_row_hover(item, True))
            row.bind("<Leave>", lambda _event, item=label: self._set_row_hover(item, False))
            row.bind("<ButtonRelease-1>", lambda _event, item=label: self._select(item))
            row.bind("<MouseWheel>", self._mousewheel)
            self._rows[label] = row

    def _draw_row(self, label: str) -> None:
        row = self._rows.get(label)
        if row is None or not row.winfo_exists():
            return
        row.delete("all")
        width = max(200, row.winfo_width())
        selected = label == self.variable.get()
        hovered = bool(getattr(row, "_rod_hovered", False))
        fill = COLORS["blue_soft"] if selected else "#f6f9fc" if hovered else "#ffffff"
        row.create_rectangle(0, 0, width, self.ROW_HEIGHT, fill=fill, outline="")
        state = self.item_states.get(label, "available")
        badge = self.item_badges.get(label)
        text_x = 52 if self.reserve_icon_space else 18
        if badge is not None:
            row.create_image(25, self.ROW_HEIGHT / 2, image=badge)
            text_x = 52
        text_fill = COLORS["blue_dark"] if selected else COLORS["text"]
        if state in {"paid", "coming_soon"} and not selected:
            text_fill = "#4b5565"
        row.create_text(
            text_x,
            self.ROW_HEIGHT / 2,
            text=label,
            anchor="w",
            fill=text_fill,
            font=semibold(11) if selected else regular(11),
        )
        if selected:
            x = width - 27
            y = self.ROW_HEIGHT / 2
            row.create_line(
                x - 6,
                y,
                x - 2,
                y + 5,
                x + 7,
                y - 6,
                fill=COLORS["blue"],
                width=3,
                capstyle="round",
                joinstyle="round",
            )

    def _set_row_hover(self, label: str, hovered: bool) -> None:
        row = self._rows.get(label)
        if row is not None:
            row._rod_hovered = hovered  # type: ignore[attr-defined]
            self._draw_row(label)

    def _move_active(self, event) -> str:
        if not self._filtered_values:
            return "break"
        direction = 1 if event.keysym == "Down" else -1
        self._active_index = (self._active_index + direction) % len(self._filtered_values)
        label = self._filtered_values[self._active_index]
        for item, row in self._rows.items():
            row._rod_hovered = item == label  # type: ignore[attr-defined]
            self._draw_row(item)
        self._list_canvas.yview_moveto(
            max(0.0, (self._active_index - 2) / max(1, len(self._filtered_values)))
        )
        return "break"

    def _choose_active(self, _event=None) -> str:
        if self._filtered_values:
            self._select(self._filtered_values[self._active_index])
        return "break"

    def _select(self, label: str) -> None:
        self.variable.set(label)
        self._close_popup()
        self.command()

    def _update_scroll_region(self, _event=None) -> None:
        if self._list_canvas.winfo_exists():
            self._list_canvas.configure(scrollregion=self._list_canvas.bbox("all"))

    def _resize_list_window(self, event) -> None:
        self._list_canvas.itemconfigure(self._list_window, width=event.width)

    def _mousewheel(self, event) -> str:
        self._list_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _update_scrollbar(self, first: str, last: str) -> None:
        self._scroll_first = float(first)
        self._scroll_last = float(last)
        self._draw_scrollbar()

    def _draw_scrollbar(self) -> None:
        if not hasattr(self, "_scrollbar") or not self._scrollbar.winfo_exists():
            return
        canvas = self._scrollbar
        canvas.delete("all")
        width = max(29, canvas.winfo_width())
        height = max(60, canvas.winfo_height())
        center = width / 2
        canvas.create_image(center, 11, image=self.scroll_up_arrow)
        canvas.create_image(center, height - 9, image=self.scroll_down_arrow)
        track_top = 24
        track_bottom = height - 20
        track_height = max(1, track_bottom - track_top)
        first = getattr(self, "_scroll_first", 0.0)
        last = getattr(self, "_scroll_last", 1.0)
        if last - first >= 0.999:
            return
        thumb_height = min(float(self.scrollbar_thumb.height()), float(track_height))
        thumb_travel = max(0.0, track_height - thumb_height)
        thumb_top = track_top + first * thumb_travel
        thumb_bottom = thumb_top + thumb_height
        self._scroll_thumb_top = thumb_top
        self._scroll_thumb_bottom = thumb_bottom
        canvas.create_image(center, thumb_top, image=self.scrollbar_thumb, anchor="n")

    def _scrollbar_press(self, event) -> str:
        height = self._scrollbar.winfo_height()
        if event.y < 24:
            self._list_canvas.yview_scroll(-1, "units")
        elif event.y > height - 20:
            self._list_canvas.yview_scroll(1, "units")
        else:
            thumb_top = getattr(self, "_scroll_thumb_top", 0.0)
            thumb_bottom = getattr(self, "_scroll_thumb_bottom", 0.0)
            if thumb_top <= event.y <= thumb_bottom:
                self._scroll_drag_offset = event.y - thumb_top
            else:
                self._scroll_drag_offset = self.scrollbar_thumb.height() / 2
            self._scrollbar_drag(event)
        return "break"

    def _scrollbar_drag(self, event) -> str:
        height = self._scrollbar.winfo_height()
        track_top = 24
        track_height = max(1, height - 44)
        thumb_height = min(float(self.scrollbar_thumb.height()), float(track_height))
        thumb_travel = max(1.0, track_height - thumb_height)
        offset = getattr(self, "_scroll_drag_offset", thumb_height / 2)
        fraction = min(1.0, max(0.0, (event.y - track_top - offset) / thumb_travel))
        self._list_canvas.yview_moveto(fraction)
        return "break"

    def _scroll_to_selected(self) -> None:
        if self.variable.get() not in self._filtered_values:
            return
        index = self._filtered_values.index(self.variable.get())
        if len(self._filtered_values) > self.max_visible_rows:
            self._list_canvas.yview_moveto(
                max(0.0, (index - self.max_visible_rows // 2) / len(self._filtered_values))
            )

    def _schedule_focus_check(self, _event=None) -> None:
        self.after(80, self._close_if_focus_left)

    def _close_if_focus_left(self) -> None:
        if self._popup is None or not self._popup.winfo_exists():
            return
        focused = self.focus_get()
        widget = focused
        while widget is not None:
            if widget == self._popup:
                return
            widget = getattr(widget, "master", None)
        self._close_popup()

    def _close_if_outside(self, event) -> None:
        if self._popup is None or not self._popup.winfo_exists():
            return
        left = self._popup.winfo_rootx()
        top = self._popup.winfo_rooty()
        right = left + self._popup.winfo_width()
        bottom = top + self._popup.winfo_height()
        if not (left <= event.x_root <= right and top <= event.y_root <= bottom):
            self._close_popup()

    def _close_popup(self) -> None:
        if self._popup_animation_job is not None:
            try:
                self.after_cancel(self._popup_animation_job)
            except tk.TclError:
                pass
            self._popup_animation_job = None
        popup = self._popup
        self._popup = None
        if popup is not None and popup.winfo_exists():
            popup.destroy()
        self._draw_trigger()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.trigger.configure(cursor="hand2" if enabled else "arrow")
        if not enabled:
            self._close_popup()
        self._draw_trigger()

    def destroy(self) -> None:
        self._close_popup()
        try:
            self.variable.trace_remove("write", self._variable_trace)
            self._search_var.trace_remove("write", self._search_trace)
        except tk.TclError:
            pass
        super().destroy()


class MacroWindow:
    """Friendly desktop shell for FischMate's independent automation engine."""

    def __init__(self, root: tk.Tk, project_root: Path, developer: bool = False):
        self.root = root
        self.project_root = project_root
        self.developer = developer
        self.show_developer_controls = False
        self.repository = ProfileRepository(project_root / "profiles")
        self.windows: list[WindowInfo] = []
        self.active_profile: dict | None = None
        self.armed_window: WindowInfo | None = None
        self.armed_profile: dict | None = None
        self.armed = False
        self.closing = False
        self.live_engine = LiveDetectionEngine(project_root, self._queue_live_status)
        self.session_widgets: list[tk.Widget] = []
        self.nav_buttons: dict[str, tk.Button] = {}
        self.pages: dict[str, tk.Frame] = {}
        self._active_page: str | None = None
        self._page_animation_job: str | None = None
        self._nav_animation_jobs: dict[str, str] = {}

        self.profile_name = tk.StringVar()
        self.profile_ids_by_label: dict[str, str] = {}
        self.profile_release_by_label: dict[str, str] = {}
        self.profile_release_messages_by_label: dict[str, str] = {}
        for name in self.repository.names():
            profile = self.repository.load(name)
            label = profile["profile"]["display_name"]
            if label in self.profile_ids_by_label:
                label = f"{label} ({name})"
            release = profile.get("release", {}) if isinstance(profile.get("release"), dict) else {}
            self.profile_ids_by_label[label] = name
            self.profile_release_by_label[label] = str(release.get("status", "available"))
            self.profile_release_messages_by_label[label] = str(release.get("message", ""))
        self.window_name = tk.StringVar()
        self.navigation_key = tk.StringVar(value="\\")
        self.state = tk.StringVar(value="Ready to fish!")
        self.state_detail = tk.StringVar(value="All set. You're good to go.")
        self.profile_summary = tk.StringVar(value="No rod profile loaded")
        self.rod_stats = tk.StringVar(value="—")
        self.mechanics = tk.StringVar(value="—")
        self.phase_value = tk.StringVar(value="IDLE")
        self.detector_value = tk.StringVar(value="IDLE")
        self.fps_value = tk.StringVar(value="0.0")
        self.bar_value = tk.StringVar(value="—")
        self.stick_value = tk.StringVar(value="—")
        self.error_value = tk.StringVar(value="—")
        self.intent_value = tk.StringVar(value="NEUTRAL")
        self.session_mode = tk.StringVar(value="automatic")

        root.title(APP_NAME)
        root.geometry("1320x900")
        root.minsize(1120, 760)
        root.configure(bg=COLORS["window"])
        root.protocol("WM_DELETE_WINDOW", self.exit)
        self._configure_style()

        self.logo_image = tk.PhotoImage(
            file=str(project_root / "assets" / "fischmate-icon-256.png")
        )
        self.sidebar_logo = self.logo_image.subsample(2, 2)
        self.ui_images = self._load_ui_images()
        self.profile_badges_by_label = {
            label: self.ui_images["paid_crown"]
            for label, status in self.profile_release_by_label.items()
            if status == "paid"
        }
        root.iconphoto(True, self.logo_image)
        windows_icon = project_root / "assets" / "fischmate.ico"
        if windows_icon.exists():
            root.iconbitmap(default=str(windows_icon))
            # Wait until Windows has created the real taskbar-owned wrapper.
            root.after(300, lambda: self._apply_windows_taskbar_icon(windows_icon))
        self.status_overlay = StatusOverlay(root)
        self._build_window()

        self.hotkeys = GlobalHotkeyWatcher(
            {
                "P": self.handle_start_hotkey,
                "O": self.reload,
                "M": self._hotkey_emergency_stop,
                "Y": self.developer_calibration,
            },
            dispatch=lambda callback: self.root.after(0, callback),
            immediate={"M"},
        )
        # P/O/M/Y are session controls, not setup-window shortcuts. Keeping the
        # watcher stopped here lets those letters pass through search and every
        # other text field normally until Start Fishing arms a session.
        self._session_hotkeys_enabled = False

        if self.profile_ids_by_label:
            preferred = next(
                (label for label, name in self.profile_ids_by_label.items() if name == "steady_rod"),
                next(iter(self.profile_ids_by_label)),
            )
            self.profile_name.set(preferred)
            self.load_profile()
        self.refresh_windows()
        self._show_page("start")

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure(
            "FischMate.TCombobox",
            padding=(10, 8),
            font=regular(11),
        )
        style.configure("FischMate.TEntry", padding=(9, 8), font=regular(11))
        style.configure("FischMate.TRadiobutton", background=COLORS["card"], font=regular(11))

    def _ui_image(self, filename: str) -> tk.PhotoImage:
        return tk.PhotoImage(
            file=str(self.project_root / "assets" / "ui" / "ready" / filename)
        )

    def _load_ui_images(self) -> dict[str, tk.PhotoImage]:
        return {
            "home_nav": self._ui_image("home-nav.png"),
            "profiles_nav": self._ui_image("profiles-nav.png"),
            "help_nav": self._ui_image("help-nav.png"),
            "advanced_nav": self._ui_image("advanced-nav.png"),
            "hero": self._ui_image("fishing-hero.png"),
            "rod_step": self._ui_image("rod-step.png"),
            "monitor_step": self._ui_image("monitor-step.png"),
            "target_step": self._ui_image("target-step.png"),
            "play_step": self._ui_image("play-step.png"),
            "binoculars_mode": self._ui_image("binoculars-mode.png"),
            "fish_mode": self._ui_image("fish-mode.png"),
            "fish_title": self._ui_image("fish-title.png"),
            "tip_lightbulb": self._ui_image("tip-lightbulb.png"),
            "care_heart": self._ui_image("care-heart.png"),
            "step_1": self._ui_image("step-1.png"),
            "step_2": self._ui_image("step-2.png"),
            "step_3": self._ui_image("step-3.png"),
            "step_4": self._ui_image("step-4.png"),
            "button_play": self._ui_image("button-play.png"),
            "button_stop": self._ui_image("button-stop.png"),
            "discord_support": self._ui_image("discord-support.png"),
            "rod_search": self._ui_image("fischmate-magnifier.png"),
            "rod_scrollbar_thumb": self._ui_image("fischmate-scrollbar-thumb.png"),
            "rod_scroll_up": self._ui_image("fischmate-scroll-up-arrow.png"),
            "rod_scroll_down": self._ui_image("fischmate-scroll-down-arrow.png"),
            "rod_trigger_default_strip": self._ui_image("rod-trigger-default-strip.png"),
            "rod_trigger_default_right": self._ui_image("rod-trigger-default-right.png"),
            "rod_trigger_open_strip": self._ui_image("rod-trigger-open-strip.png"),
            "rod_trigger_open_right": self._ui_image("rod-trigger-open-right.png"),
            "rod_trigger_arrow_down": self._ui_image("rod-trigger-arrow-down.png"),
            "rod_trigger_arrow_up": self._ui_image("rod-trigger-arrow-up.png"),
            "paid_crown": self._ui_image("paid-crown.png").subsample(52, 52),
        }

    @staticmethod
    def _label(parent: tk.Widget, text: str = "", **kwargs) -> tk.Label:
        options = {
            "text": text,
            "bg": kwargs.pop("bg", parent.cget("bg")),
            "fg": kwargs.pop("fg", COLORS["text"]),
            "font": kwargs.pop("font", regular(10)),
        }
        options.update(kwargs)
        return tk.Label(parent, **options)

    @staticmethod
    def _card(parent: tk.Widget, **kwargs) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=kwargs.pop("bg", COLORS["card"]),
            highlightbackground=kwargs.pop("border", COLORS["line"]),
            highlightthickness=1,
            bd=0,
            **kwargs,
        )

    def _build_window(self) -> None:
        shell = tk.Frame(self.root, bg=COLORS["window"])
        shell.pack(fill="both", expand=True)
        shell.grid_rowconfigure(0, weight=1)
        shell.grid_columnconfigure(1, weight=1)

        self.sidebar = tk.Frame(
            shell,
            width=258,
            bg=COLORS["sidebar"],
            highlightbackground=COLORS["line"],
            highlightthickness=1,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self._build_sidebar()

        self.content = tk.Frame(shell, bg=COLORS["window"])
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        for name in ("start", "profiles", "help", "advanced"):
            page = tk.Frame(self.content, bg=COLORS["window"])
            page.place(x=0, y=0, relwidth=1, relheight=1)
            self.pages[name] = page
        self._build_start_page(self.pages["start"])
        self._build_profiles_page(self.pages["profiles"])
        self._build_help_page(self.pages["help"])
        self._build_advanced_page(self.pages["advanced"])

    def _build_sidebar(self) -> None:
        brand = tk.Frame(self.sidebar, bg=COLORS["sidebar"])
        brand.pack(fill="x", pady=(34, 24))
        tk.Label(brand, image=self.sidebar_logo, bg=COLORS["sidebar"]).pack()
        self._label(
            brand,
            APP_NAME,
            bg=COLORS["sidebar"],
            font=bold(17),
        ).pack(pady=(12, 0))
        self._label(
            brand,
            "Easy fishing helper\nfor Roblox",
            bg=COLORS["sidebar"],
            fg=COLORS["muted"],
            font=regular(11),
            justify="center",
        ).pack(pady=(7, 0))

        nav = tk.Frame(self.sidebar, bg=COLORS["sidebar"])
        nav.pack(fill="x", padx=22)
        self._nav_button(nav, "start", self.ui_images["home_nav"], "Start")
        self._nav_button(nav, "profiles", self.ui_images["profiles_nav"], "Profiles")
        self._nav_button(nav, "help", self.ui_images["help_nav"], "Help")

        tk.Frame(self.sidebar, height=1, bg=COLORS["line"]).pack(fill="x", padx=24, pady=(24, 18))
        if self.show_developer_controls:
            advanced_nav = tk.Frame(self.sidebar, bg=COLORS["sidebar"])
            advanced_nav.pack(fill="x", padx=22)
            self._nav_button(
                advanced_nav,
                "advanced",
                self.ui_images["advanced_nav"],
                "Advanced",
                compact=True,
            )
            self._label(
                advanced_nav,
                "Diagnostics & tools",
                bg=COLORS["sidebar"],
                fg=COLORS["muted"],
                font=regular(9),
            ).pack(anchor="w", padx=52, pady=(0, 4))

        care = self._card(self.sidebar, bg=COLORS["sidebar"])
        care.pack(side="bottom", fill="x", padx=22, pady=24)
        care_title = tk.Frame(care, bg=COLORS["sidebar"])
        care_title.pack(anchor="w", padx=16, pady=(12, 4))
        tk.Label(
            care_title,
            image=self.ui_images["care_heart"],
            bg=COLORS["sidebar"],
        ).pack(side="left", padx=(0, 7))
        self._label(
            care_title,
            "Made with care",
            bg=COLORS["sidebar"],
            fg=COLORS["blue"],
            font=semibold(10),
        ).pack(side="left")
        self._label(
            care,
            "Open-source and community\nfriendly.",
            bg=COLORS["sidebar"],
            fg=COLORS["muted"],
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 14))

    def _nav_button(
        self,
        parent: tk.Widget,
        page: str,
        icon: tk.PhotoImage,
        text: str,
        compact: bool = False,
    ) -> None:
        button = tk.Button(
            parent,
            text=text,
            image=icon,
            compound="left",
            command=lambda: self._show_page(page),
            anchor="w",
            relief="flat",
            bd=0,
            bg=COLORS["sidebar"],
            fg=COLORS["muted"],
            activebackground=COLORS["blue_soft"],
            activeforeground=COLORS["blue"],
            cursor="hand2",
            font=semibold(11) if not compact else regular(11),
            padx=18,
            pady=13 if not compact else 10,
        )
        button.pack(fill="x", pady=2)
        self.nav_buttons[page] = button

    def _show_page(self, name: str) -> None:
        if name not in self.pages:
            return
        self._close_setup_dropdowns()
        if self._page_animation_job is not None:
            try:
                self.root.after_cancel(self._page_animation_job)
            except tk.TclError:
                pass
            self._page_animation_job = None
        for page in self.pages.values():
            page.place_configure(x=0)
        selected_page = self.pages[name]
        selected_page.tkraise()
        if self._active_page != name:
            selected_page.place_configure(x=16)
            self._active_page = name
            self._animate_page_slide(name, 0)
        for page, button in self.nav_buttons.items():
            selected = page == name
            job = self._nav_animation_jobs.pop(page, None)
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except tk.TclError:
                    pass
            self._animate_nav_button(
                page,
                button,
                button.cget("bg"),
                COLORS["blue_soft"] if selected else COLORS["sidebar"],
                button.cget("fg"),
                COLORS["blue"] if selected else COLORS["muted"],
                0,
            )

    def _close_setup_dropdowns(self) -> None:
        for attribute in ("run_profile_combo", "window_combo", "profile_combo"):
            dropdown = getattr(self, attribute, None)
            if dropdown is not None:
                dropdown._close_popup()

    def _animate_page_slide(self, name: str, step: int) -> None:
        if self._active_page != name or name not in self.pages:
            self._page_animation_job = None
            return
        total_steps = 7
        progress = ease_out_cubic((step + 1) / total_steps)
        self.pages[name].place_configure(x=round(16 * (1.0 - progress)))
        if step + 1 < total_steps:
            self._page_animation_job = self.root.after(12, self._animate_page_slide, name, step + 1)
        else:
            self.pages[name].place_configure(x=0)
            self._page_animation_job = None

    def _animate_nav_button(
        self,
        page: str,
        button: tk.Button,
        start_bg: str,
        target_bg: str,
        start_fg: str,
        target_fg: str,
        step: int,
    ) -> None:
        if not button.winfo_exists():
            self._nav_animation_jobs.pop(page, None)
            return
        total_steps = 6
        progress = ease_out_cubic((step + 1) / total_steps)
        button.configure(
            bg=mix_hex(start_bg, target_bg, progress),
            fg=mix_hex(start_fg, target_fg, progress),
        )
        if step + 1 < total_steps:
            self._nav_animation_jobs[page] = self.root.after(
                12,
                self._animate_nav_button,
                page,
                button,
                start_bg,
                target_bg,
                start_fg,
                target_fg,
                step + 1,
            )
        else:
            button.configure(bg=target_bg, fg=target_fg)
            self._nav_animation_jobs.pop(page, None)

    def _build_start_page(self, page: tk.Frame) -> None:
        body = tk.Frame(page, bg=COLORS["window"])
        body.pack(fill="both", expand=True, padx=25, pady=24)

        hero = self._card(body, bg=COLORS["blue_soft"], border=COLORS["blue_border"], height=132)
        hero.pack(fill="x")
        hero.pack_propagate(False)
        art = tk.Frame(hero, bg=COLORS["blue_soft"], width=275)
        art.pack(side="left", fill="y", padx=(20, 6))
        art.pack_propagate(False)
        tk.Label(art, image=self.ui_images["hero"], bg=COLORS["blue_soft"]).pack(expand=True)
        hero_text = tk.Frame(hero, bg=COLORS["blue_soft"])
        hero_text.pack(side="left", fill="both", expand=True, padx=(12, 20))
        hero_title = tk.Frame(hero_text, bg=COLORS["blue_soft"])
        hero_title.pack(anchor="w", pady=(28, 4))
        self._label(
            hero_title,
            "Let’s go fishing!",
            bg=COLORS["blue_soft"],
            font=bold(24),
        ).pack(side="left")
        tk.Label(
            hero_title,
            image=self.ui_images["fish_title"],
            bg=COLORS["blue_soft"],
        ).pack(side="left", padx=(12, 0))
        self._label(
            hero_text,
            "FischMate Macro helps you fish in Roblox automatically.",
            bg=COLORS["blue_soft"],
            fg=COLORS["muted"],
            font=regular(12),
        ).pack(anchor="w")

        self._build_rod_step(body)
        self._build_window_step(body)
        if self.show_developer_controls:
            self._build_mode_step(body)
        self._build_start_step(body)

        tip = self._card(body, bg=COLORS["blue_soft"], border=COLORS["blue_border"])
        tip.pack(fill="x", pady=(12, 0))
        tk.Label(
            tip,
            image=self.ui_images["tip_lightbulb"],
            bg=COLORS["blue_soft"],
        ).pack(side="left", padx=(18, 8), pady=9)
        self._label(
            tip,
            "Before you start:  Settings lock once FischMate is ready. Stop the session to change them.",
            bg=COLORS["blue_soft"],
            fg=COLORS["blue_dark"],
            font=regular(10),
        ).pack(side="left", pady=13)

    def _step_intro(
        self,
        card: tk.Frame,
        number: int,
        icon: tk.PhotoImage,
        title: str,
        description: str,
        width: int = 370,
    ) -> tk.Frame:
        intro = tk.Frame(card, bg=COLORS["card"], width=width)
        intro.pack(side="left", fill="y", padx=(18, 8), pady=14)
        intro.pack_propagate(False)
        tk.Label(
            intro,
            image=self.ui_images[f"step_{number}"],
            bg=COLORS["card"],
        ).pack(side="left", anchor="n")
        tk.Label(intro, image=icon, bg=COLORS["card"]).pack(
            side="left", anchor="n", padx=(15, 14)
        )
        words = tk.Frame(intro, bg=COLORS["card"])
        words.pack(side="left", fill="both", expand=True)
        self._label(
            words,
            title,
            bg=COLORS["card"],
            font=semibold(13),
        ).pack(anchor="w")
        self._label(
            words,
            description,
            bg=COLORS["card"],
            fg=COLORS["muted"],
            justify="left",
            wraplength=205,
        ).pack(anchor="w", pady=(4, 0))
        return intro

    def _build_rod_step(self, parent: tk.Widget) -> None:
        card = self._card(parent, height=120)
        card.pack(fill="x", pady=(12, 0))
        card.pack_propagate(False)
        self._step_intro(
            card,
            1,
            self.ui_images["rod_step"],
            "Pick your rod",
            "Choose which rod FischMate\nshould use.",
        )
        form = tk.Frame(card, bg=COLORS["card"])
        form.pack(side="left", fill="both", expand=True, padx=(20, 92), pady=20)
        self._label(form, "Rod", bg=COLORS["card"], font=semibold(10)).pack(anchor="w")
        self.run_profile_combo = SearchableRodDropdown(
            form,
            variable=self.profile_name,
            values=list(self.profile_ids_by_label),
            command=self.load_profile,
            search_icon=self.ui_images["rod_search"],
            scrollbar_thumb=self.ui_images["rod_scrollbar_thumb"],
            scroll_up_arrow=self.ui_images["rod_scroll_up"],
            scroll_down_arrow=self.ui_images["rod_scroll_down"],
            trigger_default_strip=self.ui_images["rod_trigger_default_strip"],
            trigger_default_right=self.ui_images["rod_trigger_default_right"],
            trigger_open_strip=self.ui_images["rod_trigger_open_strip"],
            trigger_open_right=self.ui_images["rod_trigger_open_right"],
            trigger_arrow_down=self.ui_images["rod_trigger_arrow_down"],
            trigger_arrow_up=self.ui_images["rod_trigger_arrow_up"],
            item_badges=self.profile_badges_by_label,
            item_states=self.profile_release_by_label,
        )
        self.run_profile_combo.pack(fill="x", pady=(6, 0))
        self.session_widgets.append(self.run_profile_combo)

    def _build_window_step(self, parent: tk.Widget) -> None:
        card = self._card(parent, height=112)
        card.pack(fill="x", pady=(12, 0))
        card.pack_propagate(False)
        self._step_intro(
            card,
            2,
            self.ui_images["monitor_step"],
            "Pick your Roblox window",
            "Select the Roblox window\nFischMate should watch.",
        )
        form = tk.Frame(card, bg=COLORS["card"])
        form.pack(side="left", fill="both", expand=True, padx=(20, 14), pady=20)
        self._label(form, "Game window", bg=COLORS["card"], font=semibold(10)).pack(anchor="w")
        self.window_combo = SearchableRodDropdown(
            form,
            variable=self.window_name,
            values=[],
            command=lambda: None,
            search_icon=self.ui_images["rod_search"],
            scrollbar_thumb=self.ui_images["rod_scrollbar_thumb"],
            scroll_up_arrow=self.ui_images["rod_scroll_up"],
            scroll_down_arrow=self.ui_images["rod_scroll_down"],
            trigger_default_strip=self.ui_images["rod_trigger_default_strip"],
            trigger_default_right=self.ui_images["rod_trigger_default_right"],
            trigger_open_strip=self.ui_images["rod_trigger_open_strip"],
            trigger_open_right=self.ui_images["rod_trigger_open_right"],
            trigger_arrow_down=self.ui_images["rod_trigger_arrow_down"],
            trigger_arrow_up=self.ui_images["rod_trigger_arrow_up"],
            searchable=False,
            sort_values=False,
            reserve_icon_space=False,
            placeholder="Select a Roblox window",
            empty_message="No Roblox window found",
            max_visible_rows=5,
        )
        self.window_combo.pack(fill="x", pady=(6, 0))
        self.session_widgets.append(self.window_combo)
        self.refresh_button = tk.Button(
            card,
            text="⟳  Refresh",
            command=self.refresh_windows,
            bg="white",
            fg=COLORS["text"],
            activebackground=COLORS["blue_soft"],
            relief="solid",
            bd=1,
            cursor="hand2",
            font=medium(10),
            padx=16,
            pady=9,
        )
        self.refresh_button.pack(side="right", padx=(0, 30), pady=(42, 22))
        self.session_widgets.append(self.refresh_button)

    def _mode_choice(
        self,
        parent: tk.Widget,
        value: str,
        title: str,
        description: str,
        icon: tk.PhotoImage | None = None,
    ) -> tk.Frame:
        choice = self._card(parent, border=COLORS["blue_border"] if value == "automatic" else COLORS["line"])
        choice.pack(side="left", fill="both", expand=True, padx=(0, 10) if value == "automatic" else (10, 0))
        radio = ttk.Radiobutton(
            choice,
            text=title,
            variable=self.session_mode,
            value=value,
            style="FischMate.TRadiobutton",
        )
        radio.pack(anchor="w", padx=14, pady=(11, 2))
        self._label(
            choice,
            description,
            bg=COLORS["card"],
            fg=COLORS["muted"],
            justify="left",
            wraplength=190,
        ).pack(anchor="w", padx=39, pady=(0, 10))
        if icon is not None:
            tk.Label(choice, image=icon, bg=COLORS["card"]).place(
                relx=0.87, rely=0.68, anchor="center"
            )
        self.session_widgets.append(radio)
        return choice

    def _build_mode_step(self, parent: tk.Widget) -> None:
        card = self._card(parent, height=146)
        card.pack(fill="x", pady=(12, 0))
        card.pack_propagate(False)
        self._step_intro(
            card,
            3,
            self.ui_images["target_step"],
            "Choose a mode",
            "Decide how FischMate should work.",
        )
        choices = tk.Frame(card, bg=COLORS["card"])
        choices.pack(side="left", fill="both", expand=True, padx=(18, 30), pady=14)
        self._mode_choice(
            choices,
            "automatic",
            "Automatic fishing",
            "FischMate handles casting,\nshaking, and bar control for you.",
            self.ui_images["fish_mode"],
        )
        self._mode_choice(
            choices,
            "detection",
            "Detection only",
            "FischMate only watches and\nrecords what it sees.",
            self.ui_images["binoculars_mode"],
        )

    def _build_start_step(self, parent: tk.Widget) -> None:
        number = 4 if self.show_developer_controls else 3
        card = self._card(parent, height=146)
        card.pack(fill="x", pady=(12, 0))
        card.pack_propagate(False)
        self._step_intro(
            card,
            number,
            self.ui_images["play_step"],
            "Start fishing",
            "Make sure Roblox is open,\nthen start your adventure!",
        )

        status = self._card(card, bg=COLORS["green_soft"], border=COLORS["green_border"], width=270)
        status.pack(side="left", fill="y", padx=(12, 28), pady=15)
        status.pack_propagate(False)
        self._label(
            status,
            "●  Status",
            bg=COLORS["green_soft"],
            fg=COLORS["green"],
            font=semibold(9),
        ).pack(anchor="w", padx=18, pady=(13, 4))
        self._label(
            status,
            textvariable=self.state,
            bg=COLORS["green_soft"],
            fg="#176b2a",
            font=semibold(13),
        ).pack(anchor="w", padx=18)
        self._label(
            status,
            textvariable=self.state_detail,
            bg=COLORS["green_soft"],
            fg="#41844c",
            font=regular(9),
            wraplength=230,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(5, 10))

        actions = tk.Frame(card, bg=COLORS["card"], width=265)
        actions.pack(side="right", fill="y", padx=(0, 30), pady=14)
        actions.pack_propagate(False)
        self.start_button = ActionCanvasButton(
            actions,
            text="Start Fishing",
            command=self.arm_session,
            kind="primary",
            badge_image=self.ui_images["button_play"],
            width=265,
            height=50,
        )
        self.start_button.pack(fill="x")
        ActionCanvasButton(
            actions,
            text="Emergency Stop",
            command=self.emergency_stop,
            kind="danger",
            badge_image=self.ui_images["button_stop"],
            width=265,
            height=42,
        ).pack(fill="x", pady=(10, 0))

    def _page_heading(self, page: tk.Frame, title: str, detail: str) -> tk.Frame:
        body = tk.Frame(page, bg=COLORS["window"])
        body.pack(fill="both", expand=True, padx=34, pady=32)
        self._label(body, title, font=bold(23)).pack(anchor="w")
        self._label(
            body,
            detail,
            fg=COLORS["muted"],
            font=regular(11),
        ).pack(anchor="w", pady=(5, 22))
        return body

    def _build_profiles_page(self, page: tk.Frame) -> None:
        body = self._page_heading(
            page,
            "Rod profiles",
            "Choose a rod configuration and save the navigation key you use in Roblox.",
        )
        form = self._card(body)
        form.pack(fill="x")
        form.grid_columnconfigure(0, weight=1)
        self._label(form, "Profile", bg=COLORS["card"], font=semibold(10)).grid(
            row=0, column=0, sticky="w", padx=22, pady=(20, 6)
        )
        self.profile_combo = SearchableRodDropdown(
            form,
            variable=self.profile_name,
            values=list(self.profile_ids_by_label),
            command=self.load_profile,
            search_icon=self.ui_images["rod_search"],
            scrollbar_thumb=self.ui_images["rod_scrollbar_thumb"],
            scroll_up_arrow=self.ui_images["rod_scroll_up"],
            scroll_down_arrow=self.ui_images["rod_scroll_down"],
            trigger_default_strip=self.ui_images["rod_trigger_default_strip"],
            trigger_default_right=self.ui_images["rod_trigger_default_right"],
            trigger_open_strip=self.ui_images["rod_trigger_open_strip"],
            trigger_open_right=self.ui_images["rod_trigger_open_right"],
            trigger_arrow_down=self.ui_images["rod_trigger_arrow_down"],
            trigger_arrow_up=self.ui_images["rod_trigger_arrow_up"],
            item_badges=self.profile_badges_by_label,
            item_states=self.profile_release_by_label,
        )
        self.profile_combo.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 22))
        self._label(form, "Navigation key", bg=COLORS["card"], font=semibold(10)).grid(
            row=0, column=1, sticky="w", padx=12, pady=(20, 6)
        )
        self.navigation_entry = ttk.Entry(
            form,
            textvariable=self.navigation_key,
            width=14,
            style="FischMate.TEntry",
        )
        self.navigation_entry.bind("<FocusIn>", self._navigation_key_focus)
        self.navigation_entry.bind("<ButtonRelease-1>", self._navigation_key_focus)
        self.navigation_entry.bind("<KeyPress>", self._capture_navigation_key)
        self.navigation_entry.grid(row=1, column=1, padx=12, pady=(0, 22), ipady=3)
        self.save_button = tk.Button(
            form,
            text="Save changes",
            command=self.save_profile,
            bg=COLORS["blue"],
            fg="white",
            activebackground=COLORS["blue_dark"],
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=10,
            font=semibold(10),
        )
        self.save_button.grid(row=1, column=2, padx=(10, 22), pady=(0, 22))
        self.session_widgets.extend([self.profile_combo, self.navigation_entry, self.save_button])

        details = self._card(body)
        details.pack(fill="x", pady=(16, 0))
        self._label(
            details,
            textvariable=self.profile_summary,
            bg=COLORS["card"],
            font=semibold(14),
            wraplength=820,
            justify="left",
        ).pack(anchor="w", padx=24, pady=(22, 14))
        tk.Frame(details, height=1, bg=COLORS["line"]).pack(fill="x", padx=24)
        self._label(details, "Rod statistics", bg=COLORS["card"], fg=COLORS["muted"]).pack(
            anchor="w", padx=24, pady=(16, 3)
        )
        self._label(
            details,
            textvariable=self.rod_stats,
            bg=COLORS["card"],
            font=monospace(11),
        ).pack(anchor="w", padx=24)
        self._label(details, "Enabled mechanic modules", bg=COLORS["card"], fg=COLORS["muted"]).pack(
            anchor="w", padx=24, pady=(16, 3)
        )
        self._label(
            details,
            textvariable=self.mechanics,
            bg=COLORS["card"],
            wraplength=820,
            justify="left",
        ).pack(anchor="w", padx=24, pady=(0, 22))

    def _build_help_page(self, page: tk.Frame) -> None:
        body = self._page_heading(page, "Help", "Everything needed to begin a safe FischMate session.")
        card = self._card(body)
        card.pack(fill="x")
        instructions = (
            "1. Open Roblox and equip the rod matching your selected profile.\n\n"
            "2. Choose the Roblox window on the Start page and click Start Fishing.\n\n"
            "3. FischMate focuses Roblox. Press P there to begin automation.\n\n"
            "4. Press M at any time to release all inputs and stop immediately.\n\n"
            "Settings are available before FischMate is ready. Stop the current session before changing them."
        )
        self._label(
            card,
            instructions,
            bg=COLORS["card"],
            font=regular(11),
            justify="left",
            wraplength=800,
        ).pack(anchor="w", padx=28, pady=26)

        discord_card = self._card(body, bg="#f7f8ff", border="#c7ccff")
        discord_card.pack(fill="x", pady=(16, 0))

        tk.Label(
            discord_card,
            image=self.ui_images["discord_support"],
            bg="#f7f8ff",
        ).pack(side="left", padx=(24, 18), pady=20)

        discord_copy = tk.Frame(discord_card, bg="#f7f8ff")
        discord_copy.pack(side="left", fill="both", expand=True, pady=18)
        self._label(
            discord_copy,
            "Need help? Join the FischMate Discord",
            bg="#f7f8ff",
            font=semibold(13),
        ).pack(anchor="w")
        self._label(
            discord_copy,
            "Join the server, complete verification, then tag @JustifyX or send a direct message.",
            bg="#f7f8ff",
            fg=COLORS["muted"],
            font=regular(10),
            justify="left",
            wraplength=590,
        ).pack(anchor="w", pady=(6, 0))

        tk.Button(
            discord_card,
            text="Join Discord server",
            command=self.open_discord_support,
            bg="#5865f2",
            activebackground="#4752c4",
            fg="#ffffff",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=17,
            pady=10,
            font=semibold(10),
        ).pack(side="right", padx=(18, 24), pady=28)

    def _build_advanced_page(self, page: tk.Frame) -> None:
        body = self._page_heading(
            page,
            "Advanced diagnostics",
            "Developer telemetry from screen detection and controller output.",
        )
        metrics = self._card(body)
        metrics.pack(fill="x")
        entries = [
            ("Lifecycle phase", self.phase_value),
            ("Detector", self.detector_value),
            ("Capture / processing FPS", self.fps_value),
            ("Bar center", self.bar_value),
            ("Stick center", self.stick_value),
            ("Center error", self.error_value),
            ("Controller intent", self.intent_value),
        ]
        for row, (label, value) in enumerate(entries):
            self._label(metrics, label, bg=COLORS["card"], fg=COLORS["muted"]).grid(
                row=row, column=0, sticky="w", padx=24, pady=9
            )
            self._label(
                metrics,
                textvariable=value,
                bg=COLORS["card"],
                font=monospace(11),
            ).grid(row=row, column=1, sticky="w", padx=28, pady=9)
        tk.Button(
            body,
            text="Open diagnostics folder",
            command=self.open_diagnostics,
            bg=COLORS["blue_soft"],
            fg=COLORS["blue_dark"],
            relief="solid",
            bd=1,
            padx=16,
            pady=9,
            font=semibold(10),
        ).pack(anchor="w", pady=(16, 0))

    def _selected_window(self) -> WindowInfo | None:
        index = self.window_combo.current()
        return self.windows[index] if 0 <= index < len(self.windows) else None

    def _set_session_controls_enabled(self, enabled: bool) -> None:
        combo_state = "readonly" if enabled else "disabled"
        for widget in self.session_widgets:
            try:
                if isinstance(widget, SearchableRodDropdown):
                    widget.set_enabled(enabled)
                elif isinstance(widget, ttk.Combobox):
                    widget.configure(state=combo_state)
                else:
                    widget.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass

    def _navigation_key_focus(self, _event=None) -> None:
        self.navigation_entry.after_idle(self._select_navigation_key)

    def _select_navigation_key(self) -> None:
        if (
            self.navigation_entry.winfo_exists()
            and self.navigation_entry.focus_get() == self.navigation_entry
        ):
            self.navigation_entry.selection_range(0, tk.END)
            self.navigation_entry.icursor(tk.END)

    def _capture_navigation_key(self, event) -> str | None:
        if event.keysym in {"BackSpace", "Delete"}:
            self.navigation_key.set("")
            return "break"
        # Preserve normal Ctrl/Alt shortcuts such as Ctrl+A and Alt+Tab.
        if event.state & 0x000C:
            return None
        if len(event.char) == 1 and event.char.isprintable():
            self.navigation_key.set(event.char)
            self.navigation_entry.after_idle(self._select_navigation_key)
            return "break"
        return None

    def load_profile(self) -> None:
        if self.armed or self.live_engine.running:
            return
        label = self.profile_name.get()
        try:
            profile_id = self.profile_ids_by_label[label]
            profile = self.repository.load(profile_id)
        except KeyError:
            messagebox.showerror("Profile error", "Choose a valid rod profile.", parent=self.root)
            return
        except ProfileError as exc:
            messagebox.showerror("Profile error", str(exc), parent=self.root)
            return
        release_status = self.profile_release_by_label.get(label, "available")
        if release_status in {"paid", "coming_soon"}:
            self.active_profile = None
            self._show_unavailable_profile(label, profile, release_status)
            return
        self.active_profile = profile
        self.navigation_key.set(profile["shake"]["navigation_key"])
        metadata = profile["profile"]
        self.profile_summary.set(f"{metadata['display_name']} — {metadata.get('description', '')}")
        rod = profile["rod"]
        self.rod_stats.set(
            f"Lure speed {rod['lure_speed_percent']:+g}%   ·   Control {rod['control']:g}   ·   "
            f"Resilience {rod['resilience_percent']:g}%"
        )
        self.mechanics.set("  ·  ".join(profile["mechanics"]["enabled"]))
        self.state.set("Ready to fish!")
        self.state_detail.set("All set. You're good to go.")

    def _show_unavailable_profile(self, label: str, profile: dict, status: str) -> None:
        metadata = profile["profile"]
        message = self.profile_release_messages_by_label.get(label, "")
        if status == "paid":
            title = "Full Access required"
            detail = message or "This configuration is available with the Paid version at Fischmate.com."
            stats = "Available with Full Access"
        else:
            title = "Coming soon"
            detail = message or "This configuration is still being worked on and cannot be loaded yet."
            stats = "Coming soon"
        self.navigation_key.set("\\")
        self.profile_summary.set(f"{metadata['display_name']} - {metadata.get('description', '')}")
        self.rod_stats.set(stats)
        self.mechanics.set("Locked")
        self.state.set(title)
        self.state_detail.set(detail)
        messagebox.showinfo(title, detail, parent=self.root)

    def save_profile(self) -> None:
        if self.armed or self.live_engine.running:
            messagebox.showinfo(
                "Session active",
                "Stop the current session before changing profile settings.",
                parent=self.root,
            )
            return
        label, key = self.profile_name.get(), self.navigation_key.get()
        if self.profile_release_by_label.get(label, "available") in {"paid", "coming_soon"}:
            self.load_profile()
            return
        name = self.profile_ids_by_label.get(label, "")
        if not name or len(key) != 1:
            messagebox.showerror(
                "Invalid settings",
                "Choose a profile and enter one printable navigation key.",
                parent=self.root,
            )
            return
        try:
            self.repository.save_user_settings(name, {"shake": {"navigation_key": key}})
            self.load_profile()
        except (OSError, ProfileError) as exc:
            messagebox.showerror("Could not save profile", str(exc), parent=self.root)
            return
        self.state_detail.set(f"Saved navigation key {key!r}.")

    def refresh_windows(self) -> None:
        if self.armed or self.live_engine.running:
            return
        self.windows = enumerate_roblox_windows()
        self.window_combo.set_values([window.label for window in self.windows])
        if self.windows:
            self.window_combo.current(0)
        else:
            self.window_name.set("No Roblox window found")

    def _set_session_hotkeys_enabled(self, enabled: bool) -> None:
        """Expose P/O/M/Y only while FischMate is armed or running."""
        if enabled == self._session_hotkeys_enabled:
            return
        self._session_hotkeys_enabled = enabled
        if enabled:
            self.hotkeys.start()
        else:
            self.hotkeys.stop()

    def arm_session(self) -> None:
        if self.live_engine.running:
            self.state_detail.set("FischMate is already running. Press M to stop it.")
            return
        window = self._selected_window()
        if window is None:
            self.refresh_windows()
            window = self._selected_window()
        if window is None:
            self.state.set("Roblox not found")
            self.state_detail.set("Open Roblox, refresh the window list, and try again.")
            return
        if self.active_profile is None:
            self.load_profile()
        if self.active_profile is None:
            return
        self.armed_window = window
        self.armed_profile = self.active_profile
        self.armed = True
        self._set_session_hotkeys_enabled(True)
        self._set_session_controls_enabled(False)
        self.start_button.set_enabled(False)
        self.state.set("Ready")
        self.state_detail.set("Press P inside Roblox to begin.")
        self._show_page("start")
        self.root.iconify()
        self.root.after(120, self._show_armed_overlay_and_focus)

    def _show_armed_overlay_and_focus(self) -> None:
        if not self.armed or self.armed_window is None:
            return
        self.status_overlay.show("READY", "Press P inside Roblox to begin fishing")
        if not activate_window(self.armed_window):
            self.status_overlay.update("WAITING", "Click Roblox, then press P to begin")

    def handle_start_hotkey(self) -> None:
        if self.live_engine.running:
            return
        if not self.armed or self.armed_window is None or self.armed_profile is None:
            self.state.set("Not ready")
            self.state_detail.set("Click Start Fishing before pressing P.")
            return
        if not is_foreground_window(self.armed_window):
            self.status_overlay.show("WAITING", "Focus the selected Roblox window, then press P")
            return
        automation = self.session_mode.get() == "automatic"
        self.live_engine.start(
            self.armed_window,
            self.armed_profile,
            automation_enabled=automation,
        )
        self.start_button.set_enabled(False)
        self.state.set("Starting fishing" if automation else "Starting detection")
        self.state_detail.set("Press M at any time to stop.")
        self.status_overlay.update(
            "STARTING",
            "Casting will begin now" if automation else "Detection-only capture starting",
        )

    def emergency_stop(self) -> None:
        self._set_session_hotkeys_enabled(False)
        self.live_engine.stop()
        self.armed = False
        self.armed_window = None
        self.armed_profile = None
        self._set_session_controls_enabled(True)
        self.start_button.set_enabled(True)
        self.state.set("Stopped safely")
        self.state_detail.set("All inputs were released.")
        self.status_overlay.hide()
        if not self.closing:
            self.root.deiconify()
            self.root.lift()

    def _hotkey_emergency_stop(self) -> None:
        self.live_engine.emergency_release()
        if not self.closing:
            try:
                self.root.after(0, self.emergency_stop)
            except RuntimeError:
                pass

    def reload(self) -> None:
        self.emergency_stop()
        self.load_profile()
        self.refresh_windows()
        self.state.set("Reloaded")
        self.state_detail.set("Profiles and Roblox windows were refreshed.")

    def developer_calibration(self) -> None:
        if self.developer:
            self.state_detail.set("Developer calibration marker requested. No input was emitted.")
            self._show_page("advanced")
        else:
            self.state_detail.set("Calibration tools are available through launch_debug.bat.")

    def open_diagnostics(self) -> None:
        import os

        path = self.project_root / "diagnostics"
        path.mkdir(exist_ok=True)
        os.startfile(path)

    def open_discord_support(self) -> None:
        if not webbrowser.open_new_tab(DISCORD_INVITE_URL):
            messagebox.showinfo(
                "Discord support",
                f"Open this invite in your browser:\n\n{DISCORD_INVITE_URL}",
                parent=self.root,
            )

    def _queue_live_status(self, status: LiveStatus) -> None:
        if not self.closing:
            try:
                self.root.after(0, self._apply_live_status, status)
            except RuntimeError:
                pass

    def _apply_live_status(self, status: LiveStatus) -> None:
        if self.closing:
            return
        self.phase_value.set(status.lifecycle)
        self.detector_value.set(status.detector_state)
        self.fps_value.set(f"{status.fps:.1f}")
        self.bar_value.set(self._fmt(status.bar_center))
        self.stick_value.set(self._fmt(status.stick_center))
        self.error_value.set(self._fmt(status.error_px))
        self.intent_value.set(status.command)
        if status.message:
            self._set_session_hotkeys_enabled(False)
            self.armed = False
            self.armed_window = None
            self.armed_profile = None
            self._set_session_controls_enabled(True)
            self.start_button.set_enabled(True)
            self.state.set("Session stopped")
            self.state_detail.set(status.message)
            self.status_overlay.show("STOPPED", status.message)
            self.root.deiconify()
        elif status.running:
            self.state.set("Fishing automatically" if status.input_enabled else "Watching Roblox")
            self.state_detail.set(f"{status.lifecycle} · {status.detector_state}")
            phase_details = {
                "PREPARING": "Preparing the fishing session",
                "CASTING": "Casting the selected rod",
                "SHAKING": f"Watching for SHAKE · {status.input_event} · input #{status.input_event_count}",
                "MINIGAME": f"Tracking bar and stick · intent {status.command}",
                "RESULT": "Catch finished — inputs released",
                "RECOVERY": "Waiting until the next cast is visually safe",
            }
            if (
                self.armed_profile is not None
                and self.armed_profile.get("shake", {}).get("mode") == "wait"
            ):
                phase_details["SHAKING"] = "Waiting for minigame - no shake input"
            noiseform_diagnostic = ""
            if status.diagnostic_mode:
                physical_input = "DOWN" if status.mouse_is_down else "UP"
                reason = status.command_reason or "no controller reason"
                if status.rejection_reason and status.rejection_reason != reason:
                    reason = f"{reason} / {status.rejection_reason}"
                bar_source = self._friendly_diagnostic_source(status.bar_source)
                stick_source = self._friendly_diagnostic_source(status.stick_source)
                mode_reason = status.noiseform_mode_reason or "stable"
                measured_width = (
                    "unknown"
                    if status.noiseform_measured_bar_width is None
                    else f"{status.noiseform_measured_bar_width:.0f}px"
                )
                noiseform_diagnostic = (
                    f"{status.diagnostic_mode} | Intent {status.command} | Mouse {physical_input}\n"
                    f"Reason: {reason} | Mode: {mode_reason}\n"
                    f"Bar: {bar_source} ({measured_width}, {status.noiseform_bar_width_confidence:.0%}) "
                    f"| Stick: {stick_source}"
                )
            self.status_overlay.update(
                status.lifecycle,
                phase_details.get(status.lifecycle, f"Detector {status.detector_state}"),
                noiseform_diagnostic,
            )
        else:
            self.start_button.set_enabled(False)

    @staticmethod
    def _fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.1f} px"

    @staticmethod
    def _friendly_diagnostic_source(value: str) -> str:
        if not value:
            return "missing"
        return value.removeprefix("noiseform_").replace("_", " ")

    def _apply_windows_taskbar_icon(self, icon_path: Path) -> None:
        """Assign native HICONs to Tk's outer Windows frame.

        ``iconphoto`` is enough on most systems, but some Windows/Tk builds
        continue showing pythonw.exe in the taskbar. WM_SETICON targets the
        actual native top-level frame and makes the branding deterministic.
        """
        if sys.platform != "win32":
            return
        try:
            self.root.update_idletasks()
            user32 = ctypes.windll.user32
            user32.LoadImageW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint,
            ]
            user32.LoadImageW.restype = ctypes.c_void_p
            user32.GetParent.argtypes = [ctypes.c_void_p]
            user32.GetParent.restype = ctypes.c_void_p
            user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            user32.GetAncestor.restype = ctypes.c_void_p
            user32.SendMessageW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            user32.SendMessageW.restype = ctypes.c_ssize_t
            load_flags = 0x0010  # LR_LOADFROMFILE
            image_icon = 1
            big_size = max(32, int(user32.GetSystemMetrics(11)))  # SM_CXICON
            small_size = max(16, int(user32.GetSystemMetrics(49)))  # SM_CXSMICON
            path = str(icon_path.resolve())
            big_icon = user32.LoadImageW(
                None, path, image_icon, big_size, big_size, load_flags
            )
            small_icon = user32.LoadImageW(
                None, path, image_icon, small_size, small_size, load_flags
            )
            if not big_icon or not small_icon:
                return
            client_hwnd = int(self.root.winfo_id())
            frame_hwnd = int(user32.GetParent(client_hwnd) or 0) or client_hwnd
            root_hwnd = int(user32.GetAncestor(client_hwnd, 2) or 0) or frame_hwnd
            owner_hwnd = int(user32.GetAncestor(client_hwnd, 3) or 0) or root_hwnd
            wm_seticon = 0x0080
            for hwnd in {client_hwnd, frame_hwnd, root_hwnd, owner_hwnd}:
                user32.SendMessageW(hwnd, wm_seticon, 1, big_icon)  # ICON_BIG
                user32.SendMessageW(hwnd, wm_seticon, 0, small_icon)  # ICON_SMALL
            relaunch_command = f'"{sys.executable}" -m app.main'
            icon_resource = f"{path},0"
            for hwnd in {root_hwnd, owner_hwnd}:
                set_taskbar_identity(
                    hwnd,
                    WINDOWS_APP_ID,
                    icon_resource,
                    relaunch_command,
                )
            # Keep HICON objects alive for the lifetime of the Tk window.
            self._native_icon_handles = (big_icon, small_icon)
        except (AttributeError, OSError, tk.TclError):
            # Tk's iconphoto/iconbitmap remain as safe fallbacks.
            return

    def exit(self) -> None:
        self.closing = True
        self._set_session_hotkeys_enabled(False)
        self.live_engine.stop()
        self.status_overlay.destroy()
        self.root.destroy()
