import os
import time
import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


TARGET_TEXT = "climate123"
REQUIRED_SAMPLES = 10
DATASET_FILE = "keystroke_samples.csv"
MODEL_FILE = "keystroke_model.pkl"
LOG_FILE = "attempt_logs.txt"


@dataclass
class StrokeEvent:
    key: str
    press_time: float
    release_time: Optional[float] = None


class KeystrokeAuthApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Keystroke Dynamics Authentication")
        self.root.geometry("940x640")
        self.root.resizable(False, False)

        self.feature_columns = self.build_feature_columns(TARGET_TEXT)
        self.current_events: List[StrokeEvent] = []
        self.invalid_sample = False
        self.model = None
        self.scaler = None

        self.status_var = tk.StringVar(value="Model: Not trained")
        self.count_var = tk.StringVar(
            value=f"Collected samples: 0/{REQUIRED_SAMPLES}"
        )
        self.result_var = tk.StringVar(value="Ready.")

        self.training_df = self.load_training_data()

        self.build_ui()
        self.load_model_if_exists()
        self.refresh_status()

        self.log("Application started.")
        self.log(f"Target text: {TARGET_TEXT}")
        self.log(f"Training CSV: {DATASET_FILE}")
        self.log(f"Model file: {MODEL_FILE}")

    def build_feature_columns(self, text: str) -> List[str]:
        n = len(text)

        cols = [f"dwell_{i+1}" for i in range(n)]
        cols += [f"flight_{i+1}" for i in range(n - 1)]
        cols += [
            "total_time",
            "typing_speed",
            "mean_dwell",
            "std_dwell",
            "mean_flight",
            "std_flight",
        ]

        return cols

    def build_ui(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(
            top,
            text="Keystroke Dynamics Authentication System",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            top,
            text=f"Type exactly: {TARGET_TEXT}",
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(5, 0))

        form = ttk.LabelFrame(
            self.root,
            text="Login / Sample Input",
            padding=12,
        )
        form.pack(fill="x", padx=12, pady=10)

        ttk.Label(form, text="Username:").grid(
            row=0,
            column=0,
            sticky="w",
            pady=6,
        )

        self.username_entry = ttk.Entry(
            form,
            font=("Consolas", 13),
            width=32,
        )
        self.username_entry.grid(
            row=0,
            column=1,
            sticky="w",
            pady=6,
        )

        ttk.Label(form, text="Type here:").grid(
            row=1,
            column=0,
            sticky="w",
            pady=6,
        )

        self.entry = ttk.Entry(
            form,
            font=("Consolas", 15),
            width=46,
        )
        self.entry.grid(
            row=1,
            column=1,
            sticky="w",
            pady=6,
        )

        self.entry.bind("<FocusIn>", self.start_capture)
        self.entry.bind("<KeyPress>", self.on_key_press)
        self.entry.bind("<KeyRelease>", self.on_key_release)

        btns = ttk.Frame(form)
        btns.grid(
            row=2,
            column=1,
            sticky="w",
            pady=8,
        )

        for txt, cmd in [
            ("Collect Sample", self.collect_sample),
            ("Train Model", self.train_model),
            ("Authenticate", self.authenticate),
            ("Clear", self.clear_entry),
        ]:
            ttk.Button(
                btns,
                text=txt,
                command=cmd,
            ).pack(side="left", padx=4)

        status = ttk.LabelFrame(
            self.root,
            text="Status",
            padding=12,
        )
        status.pack(fill="x", padx=12, pady=6)

        ttk.Label(
            status,
            textvariable=self.status_var,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            status,
            textvariable=self.count_var,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(
            status,
            textvariable=self.result_var,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 0))

        logf = ttk.LabelFrame(
            self.root,
            text="Activity Log",
            padding=12,
        )
        logf.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=10,
        )

        self.log_box = scrolledtext.ScrolledText(
            logf,
            height=18,
            font=("Consolas", 10),
        )
        self.log_box.pack(
            fill="both",
            expand=True,
        )

    def log(self, text: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {text}\n"

        if hasattr(self, "log_box"):
            self.log_box.insert("end", line)
            self.log_box.see("end")

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)

    def load_training_data(self) -> pd.DataFrame:
        cols = ["sample_id"] + self.feature_columns

        if os.path.exists(DATASET_FILE):
            try:
                df = pd.read_csv(DATASET_FILE)

                for c in cols:
                    if c not in df.columns:
                        df[c] = np.nan

                return df[cols].copy()

            except Exception:
                pass

        return pd.DataFrame(columns=cols)

    def save_training_data(self):
        self.training_df.to_csv(
            DATASET_FILE,
            index=False,
        )

    def load_model_if_exists(self):
        if not os.path.exists(MODEL_FILE):
            return

        try:
            with open(MODEL_FILE, "rb") as f:
                data = pickle.load(f)

            self.model = data["model"]
            self.scaler = data["scaler"]

            self.log("Existing model loaded from disk.")

        except Exception as e:
            self.log(f"Model load failed: {e}")

    def refresh_status(self):
        self.status_var.set(
            "Model: Loaded"
            if self.model is not None
            else "Model: Not trained"
        )

        self.count_var.set(
            f"Collected samples: "
            f"{len(self.training_df)}/{REQUIRED_SAMPLES}"
        )

        self.result_var.set("Ready.")

    def start_capture(self, event=None):
        self.current_events = []
        self.invalid_sample = False

    def clear_entry(self):
        self.entry.delete(0, "end")
        self.username_entry.focus_set()
        self.start_capture()
        self.result_var.set("Ready.")

    def on_key_press(self, event):
        if event.keysym in ("BackSpace", "Delete"):
            self.invalid_sample = True
            return

        ignored = {
            "Shift_L",
            "Shift_R",
            "Control_L",
            "Control_R",
            "Alt_L",
            "Alt_R",
            "Tab",
            "Escape",
            "Left",
            "Right",
            "Up",
            "Down",
            "Home",
            "End",
            "Return",
        }

        if event.keysym in ignored:
            return

        ch = event.char

        if ch and ch.isprintable():
            self.current_events.append(
                StrokeEvent(
                    ch,
                    time.perf_counter(),
                )
            )

    def on_key_release(self, event):
        ch = event.char

        if not ch or not ch.isprintable():
            return

        for ev in reversed(self.current_events):
            if ev.key == ch and ev.release_time is None:
                ev.release_time = time.perf_counter()
                break

    def sample_to_features(
        self,
        events: List[StrokeEvent],
    ) -> Optional[np.ndarray]:

        if (
            len(events) != len(TARGET_TEXT)
            or any(ev.release_time is None for ev in events)
        ):
            return None

        dwell = [
            ev.release_time - ev.press_time
            for ev in events
        ]

        flight = [
            events[i + 1].press_time
            - events[i].release_time
            for i in range(len(events) - 1)
        ]

        total = (
            events[-1].release_time
            - events[0].press_time
        )

        if total <= 0:
            return None

        vec = np.array(
            dwell
            + flight
            + [
                total,
                len(events) / total,
                float(np.mean(dwell)),
                float(np.std(dwell)),
                float(np.mean(flight)) if flight else 0.0,
                float(np.std(flight)) if flight else 0.0,
            ],
            dtype=float,
        )

        return (
            None
            if np.isnan(vec).any()
            or np.isinf(vec).any()
            else vec
        )

    def validate_input(self) -> bool:
        typed = self.entry.get().strip()

        return (
            typed == TARGET_TEXT
            and not self.invalid_sample
            and len(self.current_events) == len(TARGET_TEXT)
            and all(
                ev.release_time is not None
                for ev in self.current_events
            )
        )

    def collect_sample(self):
        if not self.validate_input():
            messagebox.showerror(
                "Invalid Sample",
                "The typed text must match exactly, "
                "and the keystroke sample must be complete.",
            )

            self.log("Training sample rejected.")
            self.clear_entry()
            return

        features = self.sample_to_features(
            self.current_events
        )

        if features is None:
            messagebox.showerror(
                "Invalid Sample",
                "Could not extract features from sample.",
            )

            self.log(
                "Training sample feature extraction failed."
            )

            self.clear_entry()
            return

        sample_id = len(self.training_df) + 1

        row = {"sample_id": sample_id}

        row.update(
            {
                c: float(v)
                for c, v in zip(
                    self.feature_columns,
                    features,
                )
            }
        )

        self.training_df = pd.concat(
            [
                self.training_df,
                pd.DataFrame([row]),
            ],
            ignore_index=True,
        )

        self.save_training_data()

        self.log(
            f"Collected training sample #{sample_id} "
            f"and saved to CSV."
        )

        self.result_var.set(
            f"Training sample #{sample_id} saved successfully."
        )

        self.refresh_status()
        self.clear_entry()

        if (
            len(self.training_df) >= REQUIRED_SAMPLES
            and self.model is None
        ):
            self.log(
                "Required samples collected. "
                "Training model automatically."
            )

            self.train_model(auto=True)

    def train_model(self, auto: bool = False):
        df = self.training_df.dropna(
            subset=self.feature_columns
        ).copy()

        if len(df) < 3:
            messagebox.showwarning(
                "Not Enough Data",
                "Collect at least 3 genuine samples before training.",
            )

            self.log(
                "Training failed: not enough samples."
            )
            return

        X = (
            df[self.feature_columns]
            .astype(float)
            .values
        )

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.model = OneClassSVM(
            kernel="rbf",
            nu=0.2,
            gamma="scale",
        )

        self.model.fit(Xs)

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "feature_columns": self.feature_columns,
                    "target_text": TARGET_TEXT,
                },
                f,
            )

        accept_rate = float(
            np.mean(
                self.model.predict(Xs) == 1
            )
            * 100
        )

        self.log(
            f"Model trained and saved to {MODEL_FILE}."
        )

        self.log(
            f"Genuine acceptance rate on training data: "
            f"{accept_rate:.2f}%"
        )

        if not auto:
            messagebox.showinfo(
                "Training Complete",
                f"Model trained successfully.\n"
                f"Genuine acceptance rate: "
                f"{accept_rate:.2f}%",
            )

        self.result_var.set(
            "Model trained successfully."
        )

        self.refresh_status()

    def authenticate(self):
        if self.model is None or self.scaler is None:
            if os.path.exists(MODEL_FILE):
                self.load_model_if_exists()

            if self.model is None or self.scaler is None:
                messagebox.showwarning(
                    "Model Missing",
                    "Train the model first before authenticating.",
                )

                self.log(
                    "Authentication blocked: model not trained."
                )
                return

        typed = self.entry.get().strip()

        if typed != TARGET_TEXT:
            self.log(
                f"Authentication failed: "
                f"typed text mismatch ({typed!r})."
            )

            self.result_var.set(
                "Authentication Failed"
            )

            messagebox.showerror(
                "Authentication Failed",
                "The entered text does not match the required text.",
            )

            self.clear_entry()
            return

        features = self.sample_to_features(
            self.current_events
        )

        if features is None or self.invalid_sample:
            self.log(
                "Authentication attempt failed: "
                "incomplete keystroke capture."
            )

            self.result_var.set(
                "Authentication Failed"
            )

            messagebox.showerror(
                "Authentication Failed",
                "Keystroke capture was incomplete or invalid.",
            )

            self.clear_entry()
            return

        Xs = self.scaler.transform(
            features.reshape(1, -1)
        )

        prediction = int(
            self.model.predict(Xs)[0]
        )

        score = float(
            self.model.decision_function(Xs)[0]
        )

        if prediction == 1:
            self.result_var.set(
                "Authentication Successful"
            )

            messagebox.showinfo(
                "Authentication Successful",
                f"User accepted.\n"
                f"Similarity score: {score:.4f}",
            )

            result = "SUCCESS"

        else:
            self.result_var.set(
                "Authentication Failed"
            )

            messagebox.showerror(
                "Authentication Failed",
                f"User rejected.\n"
                f"Similarity score: {score:.4f}",
            )

            result = "FAILED"

        self.log(
            f"Authentication attempted | "
            f"prediction={prediction} | "
            f"score={score:.6f} | "
            f"result={result}"
        )

        self.clear_entry()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = KeystrokeAuthApp(root)
    app.run()
