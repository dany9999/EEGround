import os
import h5py
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model.SupervisedClassifier import BIOTClassifier

# =========================
# CONFIG FISSA
# =========================
CKPT_PATH = "checkpoints/best-model-run6.ckpt"
THRESHOLD = 0.073144

N_FFT = 250
HOP_LENGTH = 125

FS = 250                 # Hz 
SEGMENT_SECONDS = 4.0    

APPLY_P95_NORM_FOR_INFERENCE = True

CELL_SIZE = 14
CELL_GAP = 4
CELLS_PER_ROW = 70

CHANNEL_IDXS = list(range(18))  

# euristica: seizure = almeno 20 secondi di positivi consecutivi
MIN_SEIZURE_SECONDS = 20.0
MIN_RUN = int(np.ceil(MIN_SEIZURE_SECONDS / SEGMENT_SECONDS))  # 20/4=5
# =========================


def robust_p95_norm_np(x_ct: np.ndarray) -> np.ndarray:
    p95 = np.percentile(np.abs(x_ct), 95, axis=-1, keepdims=True) + 1e-8
    return x_ct / p95


def red_yellow_from_consecutive(preds, min_run):
    """
    Rosso: run di 1 consecutivi lunga >= min_run
    Giallo: 1 non in run lunga abbastanza
    """
    preds = np.asarray(preds).astype(np.int32)
    N = len(preds)

    red = np.zeros(N, dtype=bool)

    i = 0
    while i < N:
        if preds[i] == 1:
            j = i
            while j < N and preds[j] == 1:
                j += 1
            run_len = j - i
            if run_len >= min_run:
                red[i:j] = True
            i = j
        else:
            i += 1

    yellow = (preds == 1) & (~red)
    return red, yellow


def load_ckpt_strip_dataparallel(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("net."):
            k = k[len("net."):]
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    model_sd = model.state_dict()
    loaded_ok = sum(
        1 for k, v in new_sd.items()
        if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)
    )

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Loaded keys (shape match): {loaded_ok}")
    print(f"  Missing keys             : {len(missing)}")
    print(f"  Unexpected keys          : {len(unexpected)}")

    if loaded_ok < 20:
        print("[WARNING] Hai caricato pochissime chiavi compatibili: controlla architettura/prefissi.")

    return len(missing), len(unexpected), loaded_ok


@torch.no_grad()
def predict_prob(model, x_bct: np.ndarray, device: torch.device) -> float:
    x = torch.tensor(x_bct, dtype=torch.float32, device=device)
    logits = model(x)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    return float(torch.sigmoid(logits)[0].item())


class GridDemo:
    def __init__(self, root):
        self.root = root
        root.title("EEG Seizure Detector Demo ")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_18 = None   # (N,18,T)
        self.N = 0
        self.C = 0
        self.T = 0
        self.h5_path = None

        self.model = None
        self.threshold = float(THRESHOLD)

        self.probs = None
        self.preds = None
        self.red_mask = None
        self.yellow_mask = None

        self.selected_idx = 0

        # ========= Top controls =========
        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Button(top, text="Open H5", command=self.open_h5).pack(side=tk.LEFT, padx=4)

        self.btn_run = tk.Button(top, text="Run classification", command=self.run_inference, state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, padx=4)

  

        tk.Label(top, text="View:").pack(side=tk.LEFT, padx=(14, 4))
        self.ch_var = tk.StringVar(value="ALL")
        self.ch_menu = tk.OptionMenu(top, self.ch_var, "ALL", command=self.on_channel_change)
        self.ch_menu.config(state=tk.DISABLED)
        self.ch_menu.pack(side=tk.LEFT)

        self.info = tk.StringVar(value=f"Device={self.device}")
        tk.Label(root, textvariable=self.info, anchor="w").pack(side=tk.TOP, fill=tk.X, padx=8)

        # ========= Middle: grid + scroll =========
        mid = tk.Frame(root)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.canvas = tk.Canvas(mid, bg="white", height=240)
        self.v_scroll = tk.Scrollbar(mid, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scroll = tk.Scrollbar(mid, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        mid.grid_rowconfigure(0, weight=1)
        mid.grid_columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.rect_to_idx = {}

        # ========= Bottom: plot =========
        bottom = tk.Frame(root)
        bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.fig = plt.Figure(figsize=(11, 4.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("EEG segment")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=bottom)
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ---------------- UI helpers ----------------
    def _set_channel_menu(self, C: int):
        menu = self.ch_menu["menu"]
        menu.delete(0, "end")

        menu.add_command(label="ALL", command=lambda: self.ch_var.set("ALL") or self.on_channel_change("ALL"))
        for ch in range(C):
            menu.add_command(label=str(ch), command=lambda v=ch: self.ch_var.set(str(v)) or self.on_channel_change(str(v)))

        self.ch_var.set("ALL")
        self.ch_menu.config(state=tk.NORMAL)

    # ---------------- File loading ----------------
    def open_h5(self):
        path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5 *.hdf5")])
        if not path:
            return

        try:
            with h5py.File(path, "r") as f:
                if "signals" not in f:
                    messagebox.showerror("Error", "Dataset 'signals' not found in H5.")
                    return
                data = f["signals"][:]  # (N,C,T)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if data.ndim != 3:
            messagebox.showerror("Error", f"'signals' must be (N,C,T). Found: {data.shape}")
            return

        N, Cfull, T = data.shape

        if len(CHANNEL_IDXS) != 18:
            messagebox.showerror("Error", f"CHANNEL_IDXS deve avere 18 indici. Ora: {len(CHANNEL_IDXS)}")
            return
        if max(CHANNEL_IDXS) >= Cfull or min(CHANNEL_IDXS) < 0:
            messagebox.showerror("Error", f"CHANNEL_IDXS fuori range. File ha C={Cfull}, idx max={max(CHANNEL_IDXS)}")
            return

        data_18 = data[:, CHANNEL_IDXS, :]  # (N,18,T)

        # sanity check durata segmento
        seg_sec = T / float(FS)
        if abs(seg_sec - SEGMENT_SECONDS) > 0.25:
            print(f"[WARNING] Segment length ~= {seg_sec:.3f}s (T={T}, FS={FS}). Atteso ~{SEGMENT_SECONDS}s.")

        self.h5_path = path
        self.data_18 = data_18
        self.N, self.C, self.T = data_18.shape  # C=18

        # modello
        self.model = BIOTClassifier(
            n_channels=self.C,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        ).to(self.device)

        if not os.path.exists(CKPT_PATH):
            messagebox.showerror("Missing CKPT", f"Checkpoint not found:\n{CKPT_PATH}")
            return

        missing, unexpected, loaded_ok = load_ckpt_strip_dataparallel(self.model, CKPT_PATH, self.device)
        self.model.eval()

        # reset
        self.probs = None
        self.preds = None
        self.red_mask = None
        self.yellow_mask = None
        self.selected_idx = 0

        self._set_channel_menu(self.C)
        self.btn_run.config(state=tk.NORMAL)

        self.info.set(f"Loaded {os.path.basename(self.h5_path)}")

        self.draw_grid(initial=True)
        self.plot_segment(0, self.ch_var.get())

    # ---------------- Inference ----------------
    def run_inference(self):
        if self.data_18 is None or self.model is None:
            return

        probs = np.zeros(self.N, dtype=np.float32)
        preds = np.zeros(self.N, dtype=np.int64)

        for i in range(self.N):
            seg = self.data_18[i]  # (18,T)

            seg_in = seg
            if APPLY_P95_NORM_FOR_INFERENCE:
                seg_in = robust_p95_norm_np(seg)

            x_bct = seg_in[None, :, :]  # (1,18,T)
            p = predict_prob(self.model, x_bct, self.device)
            probs[i] = p
            preds[i] = int(p >= self.threshold)

        self.probs = probs
        self.preds = preds

        # euristica: rosso/giallo (solo run consecutive >= MIN_RUN)
        self.red_mask, self.yellow_mask = red_yellow_from_consecutive(self.preds, min_run=MIN_RUN)

        self.draw_grid(initial=False)
        self.plot_segment(self.selected_idx, self.ch_var.get())

        n_pos = int(preds.sum())
        n_red = int(self.red_mask.sum())
        self.info.set(
            f"EEG_file: {os.path.basename(self.h5_path)} | pos={n_pos} ({n_pos/self.N:.2%})"
        )

    # ---------------- Grid ----------------
    def draw_grid(self, initial: bool):
        self.canvas.delete("all")
        self.rect_to_idx = {}

        rows = (self.N + CELLS_PER_ROW - 1) // CELLS_PER_ROW
        total_w = CELLS_PER_ROW * (CELL_SIZE + CELL_GAP) + CELL_GAP
        total_h = rows * (CELL_SIZE + CELL_GAP) + CELL_GAP

        for idx in range(self.N):
            r = idx // CELLS_PER_ROW
            c = idx % CELLS_PER_ROW

            x0 = CELL_GAP + c * (CELL_SIZE + CELL_GAP)
            y0 = CELL_GAP + r * (CELL_SIZE + CELL_GAP)
            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE

            if initial or self.preds is None:
                fill = "#DDDDDD"
            else:
                # Rosso = run lunga (seizure), Giallo = sospetto, Grigio = no
                if self.red_mask is not None and self.red_mask[idx]:
                    fill = "#E74C3C"  # rosso
                elif self.yellow_mask is not None and self.yellow_mask[idx]:
                    fill = "#F1C40F"  # giallo
                else:
                    fill = "#D0D0D0"  # grigio

            rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#888888", width=1)
            self.rect_to_idx[rect] = idx

        self.highlight_selected(self.selected_idx)
        self.canvas.configure(scrollregion=(0, 0, total_w, total_h))

    def highlight_selected(self, idx: int):
        for rect, ridx in self.rect_to_idx.items():
            if ridx == idx:
                self.canvas.itemconfig(rect, outline="#1F77B4", width=3)
            else:
                self.canvas.itemconfig(rect, outline="#888888", width=1)

    def on_canvas_click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        hit = self.canvas.find_closest(x, y)
        if not hit:
            return
        rect_id = hit[0]
        if rect_id not in self.rect_to_idx:
            return

        idx = self.rect_to_idx[rect_id]
        self.selected_idx = idx
        self.highlight_selected(idx)
        self.plot_segment(idx, self.ch_var.get())

    def on_channel_change(self, _=None):
        if self.data_18 is None:
            return
        self.plot_segment(self.selected_idx, self.ch_var.get())

    # ---------------- Plot ----------------
    def plot_segment(self, idx: int, ch_choice):
        if self.data_18 is None:
            return

        idx = int(np.clip(idx, 0, self.N - 1))
        seg = self.data_18[idx]  # (18,T)

        # asse tempo assoluto in secondi (senza gap/bug)
        seg_len_s = self.T / float(FS)
        t0 = idx * seg_len_s
        t = np.arange(self.T) / float(FS) + t0
        t1 = t0 + seg_len_s

        self.ax.clear()
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        # info pred (se disponibile)
       
        if self.probs is not None:
            p = float(self.probs[idx])

            

        if str(ch_choice).upper() == "ALL":
            scale = np.percentile(np.abs(seg), 95)
            scale = max(scale, 1e-6)
            step = 3.5 * scale

            offsets = np.arange(self.C)[::-1] * step
            for ch in range(self.C):
                y = seg[ch] + offsets[ch]
                self.ax.plot(t, y, linewidth=0.8)

            self.ax.set_yticks(offsets)
            self.ax.set_yticklabels([f"Ch {ch}" for ch in range(self.C)][::-1])

            title = f"Segment {idx}/{self.N-1} | {t0:.2f}s → {t1:.2f}s | ALL channels"
            self.ax.set_title(title)
        else:
            ch = int(ch_choice)
            ch = int(np.clip(ch, 0, self.C - 1))
            y = seg[ch]
            self.ax.plot(t, y, linewidth=1.0)
            title = f"Segment {idx}/{self.N-1} | {t0:.2f}s → {t1:.2f}s | Channel {ch}"
            self.ax.set_title(title)

        self.ax.set_xlim(t0, t1)
        self.fig.tight_layout()
        self.fig_canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = GridDemo(root)
    root.mainloop()