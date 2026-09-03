from __future__ import annotations

import tkinter as tk

from app.gui.fonts import medium, regular, semibold

from app.branding import APP_NAME


class StatusOverlay:
    """Small top-left status window kept away from the fishing detector ROI."""

    def __init__(self, root: tk.Tk):
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.geometry("340x118+18+90")
        self.window.configure(bg="#123d5a", highlightbackground="#4aa3cf", highlightthickness=1)
        self.title = tk.Label(
            self.window,
            text=APP_NAME,
            bg="#123d5a",
            fg="#ffffff",
            font=semibold(11),
            anchor="w",
        )
        self.title.pack(fill="x", padx=12, pady=(9, 0))
        self.phase = tk.Label(
            self.window,
            text="READY",
            bg="#123d5a",
            fg="#74d5ff",
            font=semibold(13),
            anchor="w",
        )
        self.phase.pack(fill="x", padx=12, pady=(2, 0))
        self.detail = tk.Label(
            self.window,
            text="Press P inside Roblox to begin fishing",
            bg="#123d5a",
            fg="#d9e8f0",
            font=medium(9),
            anchor="w",
        )
        self.detail.pack(fill="x", padx=12, pady=(1, 0))
        self.diagnostic = tk.Label(
            self.window,
            text="",
            bg="#0d3048",
            fg="#e9f7ff",
            font=medium(8),
            anchor="w",
            justify="left",
            wraplength=314,
            padx=7,
            pady=4,
        )
        self.safety = tk.Label(
            self.window,
            text="M  Emergency stop",
            bg="#123d5a",
            fg="#ffcf70",
            font=regular(8),
            anchor="w",
        )
        self.safety.pack(fill="x", padx=12, pady=(2, 8))

    def show(self, phase: str, detail: str) -> None:
        self.update(phase, detail)
        self.window.deiconify()
        self.window.lift()

    def update(self, phase: str, detail: str, diagnostic: str = "") -> None:
        self.phase.configure(text=phase)
        self.detail.configure(text=detail)
        if diagnostic:
            self.diagnostic.configure(text=diagnostic)
            if not self.diagnostic.winfo_manager():
                self.diagnostic.pack(
                    fill="x",
                    padx=12,
                    pady=(4, 0),
                    before=self.safety,
                )
            self.window.geometry("340x174+18+90")
        else:
            if self.diagnostic.winfo_manager():
                self.diagnostic.pack_forget()
            self.window.geometry("340x118+18+90")

    def hide(self) -> None:
        self.window.withdraw()

    def destroy(self) -> None:
        self.window.destroy()
