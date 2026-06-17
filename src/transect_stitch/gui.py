"""A small Tkinter GUI for Transect Stitch.

Two workflows:

* **Single mosaic** — pick a group of images and stitch them into one mosaic.
* **Batch** — stitch a long run in groups of N (e.g. one mosaic every 40
  photos), writing one output file per group into a folder.

Tkinter ships with CPython, so the GUI needs no extra dependencies beyond the
imaging stack already required for stitching. The actual stitch runs on a worker
thread so the window stays responsive, posting progress back through a queue.

Launch with ``transect-stitch-gui`` or ``python -m transect_stitch.gui``.
"""

from __future__ import annotations

import queue
import tempfile
import threading
import traceback
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .metadata import (
    ORDER_AUTO,
    ORDER_FILENAME,
    ORDER_GPS,
    ORDER_TIME,
    FrameInfo,
    chunk_frames,
    order_frames,
    read_frame_info,
    stride_frames,
)

IMAGE_FILETYPES = [
    ("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
    ("All files", "*.*"),
]
VIDEO_FILETYPES = [
    ("Videos", "*.mp4 *.mov *.avi *.mkv *.mts *.m2ts *.m4v"),
    ("All files", "*.*"),
]


class StitchApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Transect Stitch")
        root.minsize(720, 560)

        # path -> FrameInfo cache so EXIF is read once per image
        self._info_cache: dict[Path, FrameInfo] = {}
        self._paths: List[Path] = []  # as added (pre-ordering)
        self._ordered: List[FrameInfo] = []  # current display/stitch order
        self._temp_dirs: List[Path] = []  # temp dirs for extracted video frames
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        self._build_widgets()
        self.root.after(100, self._drain_events)

    # ------------------------------------------------------------------ UI
    def _build_widgets(self) -> None:
        pad = {"padx": 6, "pady": 4}

        # --- source buttons ---
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)
        ttk.Button(top, text="Add Images…", command=self._add_images).pack(side="left")
        ttk.Button(top, text="Add Folder…", command=self._add_folder).pack(side="left", padx=4)
        ttk.Button(top, text="Add Video…", command=self._add_video).pack(side="left")
        ttk.Button(top, text="Remove Selected", command=self._remove_selected).pack(side="left", padx=4)
        ttk.Button(top, text="Clear", command=self._clear).pack(side="left")
        ttk.Label(top, text="Order:").pack(side="left", padx=(16, 2))
        self.order_var = tk.StringVar(value=ORDER_AUTO)
        order_box = ttk.Combobox(
            top,
            textvariable=self.order_var,
            values=[ORDER_AUTO, ORDER_FILENAME, ORDER_TIME, ORDER_GPS],
            state="readonly",
            width=10,
        )
        order_box.pack(side="left")
        order_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_list())

        # --- image list ---
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, **pad)
        self.listbox = tk.Listbox(mid, selectmode="extended", activestyle="none")
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self.listbox.yview)
        sb.pack(side="left", fill="y")
        self.listbox.config(yscrollcommand=sb.set)

        # --- options ---
        opt = ttk.LabelFrame(self.root, text="Options")
        opt.pack(fill="x", **pad)
        ttk.Label(opt, text="Detector:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.detector_var = tk.StringVar(value="orb")
        ttk.Combobox(
            opt, textvariable=self.detector_var, values=["orb", "sift"],
            state="readonly", width=6,
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(opt, text="Blend:").grid(row=0, column=2, sticky="w", padx=4)
        self.blend_var = tk.StringVar(value="feather")
        ttk.Combobox(
            opt, textvariable=self.blend_var, values=["feather", "overwrite"],
            state="readonly", width=9,
        ).grid(row=0, column=3, sticky="w")
        ttk.Label(opt, text="Max dim (px, 0=full):").grid(row=0, column=4, sticky="w", padx=4)
        self.maxdim_var = tk.StringVar(value="0")
        ttk.Entry(opt, textvariable=self.maxdim_var, width=7).grid(row=0, column=5, sticky="w")
        ttk.Label(opt, text="Use every Nth image:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.stride_var = tk.IntVar(value=1)
        ttk.Spinbox(opt, from_=1, to=100, textvariable=self.stride_var, width=5,
                    command=self._refresh_list).grid(row=1, column=1, sticky="w")
        ttk.Label(opt, text="Undistort (k1, 0=off):").grid(row=1, column=2, sticky="w", padx=4)
        self.undistort_var = tk.StringVar(value="0.0")
        ttk.Entry(opt, textvariable=self.undistort_var, width=7).grid(row=1, column=3, sticky="w")
        self.clahe_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Enhance contrast (CLAHE)", variable=self.clahe_var).grid(
            row=1, column=4, columnspan=2, sticky="w", padx=4)
        ttk.Label(opt, text="Video: every Nth frame:").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.video_stride_var = tk.IntVar(value=5)
        ttk.Spinbox(opt, from_=1, to=300, textvariable=self.video_stride_var, width=5).grid(
            row=2, column=1, sticky="w")
        ttk.Label(opt, text="(5 = 6 fps from 30 fps video; lower = more overlap)",
                  foreground="gray").grid(row=2, column=2, columnspan=4, sticky="w", padx=4)

        # --- mode ---
        mode = ttk.LabelFrame(self.root, text="Mode")
        mode.pack(fill="x", **pad)
        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            mode, text="Single mosaic from all images", value="single",
            variable=self.mode_var, command=self._update_mode,
        ).grid(row=0, column=0, sticky="w", padx=4, pady=2, columnspan=3)
        ttk.Radiobutton(
            mode, text="Batch: one mosaic every", value="batch",
            variable=self.mode_var, command=self._update_mode,
        ).grid(row=1, column=0, sticky="w", padx=4)
        self.batch_size_var = tk.IntVar(value=40)
        self.batch_spin = ttk.Spinbox(mode, from_=2, to=100000, textvariable=self.batch_size_var,
                                      width=6)
        self.batch_spin.grid(row=1, column=1, sticky="w")
        ttk.Label(mode, text="images").grid(row=1, column=2, sticky="w")

        # --- output ---
        out = ttk.Frame(self.root)
        out.pack(fill="x", **pad)
        self.out_btn = ttk.Button(out, text="Output file…", command=self._choose_output)
        self.out_btn.pack(side="left")
        self.out_var = tk.StringVar(value="")
        ttk.Entry(out, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=6)

        # --- action + progress ---
        act = ttk.Frame(self.root)
        act.pack(fill="x", **pad)
        self.stitch_btn = ttk.Button(act, text="Stitch", command=self._start_stitch)
        self.stitch_btn.pack(side="left")
        self.progress = ttk.Progressbar(act, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=6)

        # --- log ---
        self.log = tk.Text(self.root, height=8, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=False, padx=6, pady=(0, 6))

        self._update_mode()

    # --------------------------------------------------------------- actions
    def _add_images(self) -> None:
        files = filedialog.askopenfilenames(title="Select images", filetypes=IMAGE_FILETYPES)
        self._add_paths([Path(f) for f in files])

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select a folder of images")
        if not folder:
            return
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        self._add_paths([p for p in Path(folder).iterdir() if p.suffix.lower() in exts])

    def _add_video(self) -> None:
        path = filedialog.askopenfilename(title="Select a video file",
                                          filetypes=VIDEO_FILETYPES)
        if not path:
            return
        if self._worker and self._worker.is_alive():
            messagebox.showwarning("Transect Stitch", "Wait for the current job to finish first.")
            return
        stride = max(1, self.video_stride_var.get())
        video_path = Path(path)
        self._log(f"Extracting every {stride}th frame from {video_path.name} …")
        self.stitch_btn.config(state="disabled")
        self._worker = threading.Thread(
            target=self._extract_video_worker, args=(video_path, stride), daemon=True
        )
        self._worker.start()

    def _extract_video_worker(self, video_path: Path, stride: int) -> None:
        from .video import extract_frames
        tmp = Path(tempfile.mkdtemp(prefix="transect_stitch_"))
        self._temp_dirs.append(tmp)

        def progress(idx, total, msg):
            self._post("log_replace", msg)

        try:
            infos = extract_frames(video_path, tmp, stride=stride, progress=progress)
        except Exception as exc:
            self._post("log", f"  error extracting video: {exc}")
            self._post("extract_done", [])
            return
        self._post("log", f"  extracted {len(infos)} frames from {video_path.name}")
        self._post("extract_done", infos)

    def _add_paths(self, paths: List[Path]) -> None:
        existing = set(self._paths)
        added = 0
        for p in paths:
            if p not in existing:
                self._paths.append(p)
                existing.add(p)
                added += 1
        if added:
            self._log(f"Added {added} image(s).")
            self._refresh_list()

    def _remove_selected(self) -> None:
        # listbox is in display (ordered) order; map back to paths and drop them.
        sel = list(self.listbox.curselection())
        if not sel:
            return
        drop = {self._ordered[i].path for i in sel}
        self._paths = [p for p in self._paths if p not in drop]
        self._refresh_list()

    def _clear(self) -> None:
        self._paths.clear()
        self._refresh_list()

    def _info_for(self, path: Path) -> FrameInfo:
        info = self._info_cache.get(path)
        if info is None:
            info = read_frame_info(path)
            self._info_cache[path] = info
        return info

    def _refresh_list(self) -> None:
        infos = [self._info_for(p) for p in self._paths]
        try:
            ordered = order_frames(infos, self.order_var.get())
        except ValueError:
            # requested metadata missing for some frames; fall back to filename
            ordered = order_frames(infos, ORDER_FILENAME)
            if self._paths:
                self._log(f"order='{self.order_var.get()}' unavailable for all frames; "
                          "showing filename order instead.")
        ordered = stride_frames(ordered, max(1, self.stride_var.get()))
        self._ordered = ordered

        self.listbox.delete(0, "end")
        for i, f in enumerate(ordered):
            tag = f" [{f.timestamp_source}]" if f.timestamp_source != "none" else ""
            self.listbox.insert("end", f"{i + 1:>4}  {f.path.name}{tag}")
        self._set_status_title()

    def _set_status_title(self) -> None:
        n = len(self._ordered)
        if self.mode_var.get() == "batch" and n:
            size = max(2, self.batch_size_var.get())
            groups = len(chunk_frames(self._ordered, size))
            self.root.title(f"Transect Stitch — {n} images → {groups} mosaic(s)")
        else:
            self.root.title(f"Transect Stitch — {n} image(s)")

    def _update_mode(self) -> None:
        batch = self.mode_var.get() == "batch"
        self.out_btn.config(text="Output folder…" if batch else "Output file…")
        self.out_var.set("")
        self.batch_spin.config(state="normal" if batch else "disabled")
        self._set_status_title()

    def _choose_output(self) -> None:
        if self.mode_var.get() == "batch":
            folder = filedialog.askdirectory(title="Choose output folder")
            if folder:
                self.out_var.set(folder)
        else:
            path = filedialog.asksaveasfilename(
                title="Save mosaic as", defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tif")],
            )
            if path:
                self.out_var.set(path)

    # --------------------------------------------------------------- stitching
    def _start_stitch(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        if not self._ordered:
            messagebox.showwarning("Transect Stitch", "Add some images first.")
            return
        out = self.out_var.get().strip()
        if not out:
            messagebox.showwarning("Transect Stitch", "Choose an output location first.")
            return

        try:
            max_dim = int(self.maxdim_var.get())
        except ValueError:
            messagebox.showerror("Transect Stitch", "Max dim must be a whole number.")
            return
        try:
            undistort = float(self.undistort_var.get())
        except ValueError:
            messagebox.showerror("Transect Stitch", "Undistort (k1) must be a number.")
            return

        # Build config + job description on the main thread, then hand to worker.
        from .stitch import StitchConfig

        cfg = StitchConfig(
            detector=self.detector_var.get(),
            blend=self.blend_var.get(),
            max_dim=max_dim,
            undistort=undistort,
            clahe=self.clahe_var.get(),
        )
        batch = self.mode_var.get() == "batch"
        if batch:
            groups = chunk_frames(self._ordered, max(2, self.batch_size_var.get()))
        else:
            groups = [list(self._ordered)]

        self.stitch_btn.config(state="disabled")
        self.progress.config(maximum=len(groups), value=0)
        self._log(f"Starting: {len(groups)} mosaic(s)…")
        self._worker = threading.Thread(
            target=self._run_jobs, args=(groups, cfg, Path(out), batch), daemon=True
        )
        self._worker.start()

    def _run_jobs(self, groups, cfg, out: Path, batch: bool) -> None:
        """Worker thread: stitch each group, posting events back to the UI."""
        import cv2

        from .stitch import StitchError, stitch_frames

        done = 0
        failures = 0
        for gi, group in enumerate(groups):
            label = f"mosaic {gi + 1}/{len(groups)}" if batch else "mosaic"
            self._post("log", f"Stitching {label} ({len(group)} images)…")
            try:
                mosaic = stitch_frames(group, cfg)
            except StitchError as exc:
                failures += 1
                self._post("log", f"  skipped {label}: {exc}")
                self._post("progress", gi + 1)
                continue
            except Exception:  # pragma: no cover - defensive
                failures += 1
                self._post("log", f"  error on {label}:\n{traceback.format_exc()}")
                self._post("progress", gi + 1)
                continue

            if batch:
                out.mkdir(parents=True, exist_ok=True)
                first = group[0].path.stem
                last = group[-1].path.stem
                dest = out / f"mosaic_{gi + 1:03d}_{first}_to_{last}.jpg"
            else:
                dest = out
                if dest.parent and not dest.parent.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)

            if cv2.imwrite(str(dest), mosaic):
                h, w = mosaic.shape[:2]
                done += 1
                self._post("log", f"  wrote {dest.name} ({w}x{h})")
            else:
                failures += 1
                self._post("log", f"  failed to write {dest}")
            self._post("progress", gi + 1)

        self._post("done", (done, failures))

    # --------------------------------------------------------------- plumbing
    def _post(self, kind: str, payload) -> None:
        self._events.put((kind, payload))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "log_replace":
                    # overwrite the last line for progress updates (extraction %)
                    self.log.config(state="normal")
                    self.log.delete("end-2l", "end-1c")
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                    self.log.config(state="disabled")
                elif kind == "progress":
                    self.progress.config(value=payload)
                elif kind == "done":
                    done, failures = payload
                    self._log(f"Finished: {done} mosaic(s) written, {failures} skipped/failed.")
                    self.stitch_btn.config(state="normal")
                elif kind == "extract_done":
                    infos = payload
                    if infos:
                        for info in infos:
                            self._info_cache[info.path] = info
                        new_paths = [i.path for i in infos]
                        self._add_paths(new_paths)
                    self.stitch_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _log(self, message: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


def main() -> int:
    root = tk.Tk()
    StitchApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
