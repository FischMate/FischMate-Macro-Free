# FischMate Macro

FischMate is an open-source, screen-based fishing automation project. It uses ordinary
window capture and input APIs only; it does not inject into Roblox, inspect
process memory, or attempt to hide itself.

This is an independent implementation. Existing macros are used only to
inventory expected user-facing capabilities and file-interoperability needs.
Algorithms, state handling, controller behavior, diagnostics, configuration
schema, naming, and interface design are developed from observable gameplay,
recordings, and this project's requirements rather than translated source.

The architecture intentionally separates four responsibilities:

1. capture supplies a Roblox-window-relative frame;
2. detection reports visible facts and confidence;
3. strategy/controller chooses a target and command;
4. executor applies an interruptible input command.

Recorded-video replay and live capture both call the same `MacroPipeline` and
the same detector instances.

## Current milestone

The guided launcher provides automatic fishing and, during development, a
detection-only mode. Choose a rod and Roblox client, click **Start Fishing**, then press global `P` from
inside Roblox to begin. Once FischMate shows **Ready**, input is accepted only while that selected Roblox window
remains foreground. Losing focus or pressing global `M` disables future commands
before releasing every held input.

The public release hides developer calibration and diagnostics controls.

## Launch

Double-click `FischMate.vbs` for the clean, console-free launcher. FischMate
remains ordinary inspectable Python source and uses its FischMate taskbar icon.

## Typography

FischMate bundles the official Inter 4.1 Regular, Medium, SemiBold, and Bold
desktop faces and registers them privately for the running process. Users do not
need to install fonts system-wide. Inter is distributed under the SIL Open Font
License; its license is included at `assets/fonts/inter/LICENSE.txt`.
