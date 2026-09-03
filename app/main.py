from __future__ import annotations

import argparse
import datetime as dt
import tkinter as tk
from pathlib import Path

from app.branding import APP_NAME
from app.config.profiles import ProfileRepository
from app.gui.main_window import MacroWindow
from app.gui.fonts import load_bundled_fonts
from app.replay import replay_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} — independent screen automation")
    parser.add_argument("--developer", action="store_true")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--profile", default="steady_rod")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = ProfileRepository(PROJECT_ROOT / "profiles")
    if args.replay:
        profile = repository.load(args.profile)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = args.output or PROJECT_ROOT / "diagnostics" / f"replay_{stamp}"
        replay_video(
            args.replay,
            profile,
            output,
            render_overlay=not args.no_overlay,
            start_s=args.start,
            duration_s=args.duration,
        )
        return
    load_bundled_fonts(PROJECT_ROOT)
    root = tk.Tk()
    MacroWindow(root, PROJECT_ROOT, developer=args.developer)
    root.mainloop()


if __name__ == "__main__":
    main()
