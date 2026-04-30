import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw

class StrideAnnotator:
    def __init__(self, root, folder_path, save_json_path=None, max_width=1400, max_height=850):
        self.root = root
        self.root.title("Stride Annotation Viewer")

        self.folder_path = folder_path
        self.max_width = max_width
        self.max_height = max_height

        self.image_files = self.load_images(folder_path)
        if not self.image_files:
            raise ValueError(f"No image files found in folder: {folder_path}")

        self.current_index = 0
        self.tk_img = None

        if save_json_path is None:
            self.save_json_path = os.path.join(folder_path, "a_manual_stride_events.json")
        else:
            self.save_json_path = save_json_path

        self.annotations = {
            "left_events": [],
            "right_events": []
        }

        self.load_existing_annotations()

        # -------- image area --------
        self.image_label = tk.Label(root, bg="black")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        # -------- info --------
        self.info_label = tk.Label(root, text="", font=("Arial", 12))
        self.info_label.pack(pady=5)

        self.help_label = tk.Label(
            root,
            text=(
                "Left/Right arrows: prev/next frame | "
                "A: mark LEFT event | D: mark RIGHT event | "
                "Z: delete nearest LEFT | C: delete nearest RIGHT | "
                "X: clear all events | "
                "S: save | G: export stride segments"
            ),
            font=("Arial", 11)
        )
        self.help_label.pack(pady=5)

        # -------- buttons --------
        control_frame = tk.Frame(root)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.prev_button = tk.Button(control_frame, text="<< Prev", command=self.prev_frame, width=12)
        self.prev_button.pack(side="left")

        self.next_button = tk.Button(control_frame, text="Next >>", command=self.next_frame, width=12)
        self.next_button.pack(side="right")

        self.save_button = tk.Button(control_frame, text="Save", command=self.save_annotations, width=12)
        self.save_button.pack(side="right", padx=5)

        self.export_button = tk.Button(control_frame, text="Export Strides", command=self.export_stride_segments, width=16)
        self.export_button.pack(side="right", padx=5)

        self.clear_button = tk.Button(control_frame, text="Clear Events", command=self.clear_events, width=14)
        self.clear_button.pack(side="right", padx=5)

        # -------- slider --------
        self.slider = tk.Scale(
            root,
            from_=0,
            to=len(self.image_files) - 1,
            orient="horizontal",
            command=self.on_slider_move,
            length=1200
        )
        self.slider.pack(fill="x", padx=10, pady=10)

        # -------- jump box --------
        jump_frame = tk.Frame(root)
        jump_frame.pack(pady=5)

        tk.Label(jump_frame, text="Jump to frame index: ").pack(side="left")
        self.jump_entry = tk.Entry(jump_frame, width=10)
        self.jump_entry.pack(side="left")
        tk.Button(jump_frame, text="Go", command=self.jump_to_frame).pack(side="left", padx=5)

        # -------- key bindings --------
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("a", lambda e: self.mark_left_event())
        self.root.bind("d", lambda e: self.mark_right_event())
        self.root.bind("z", lambda e: self.delete_nearest_left())
        self.root.bind("c", lambda e: self.delete_nearest_right())
        self.root.bind("x", lambda e: self.clear_events())
        self.root.bind("s", lambda e: self.save_annotations())
        self.root.bind("g", lambda e: self.export_stride_segments())

        self.show_frame(0)

    def load_images(self, folder_path):
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(valid_exts)
        ]
        files.sort()
        return files

    def load_existing_annotations(self):
        if os.path.exists(self.save_json_path):
            try:
                with open(self.save_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.annotations["left_events"] = sorted(set(data.get("left_events", [])))
                self.annotations["right_events"] = sorted(set(data.get("right_events", [])))
                print(f"Loaded existing annotations from {self.save_json_path}")
            except Exception as e:
                print(f"Failed to load existing annotation file: {e}")

    def resize_image(self, img):
        w, h = img.size
        scale = min(self.max_width / w, self.max_height / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def draw_overlay(self, img, frame_idx, filename):
        draw = ImageDraw.Draw(img)

        text_lines = [
            f"Frame index: {frame_idx}",
            f"File: {filename}",
            f"LEFT events: {len(self.annotations['left_events'])}",
            f"RIGHT events: {len(self.annotations['right_events'])}",
        ]

        if frame_idx in self.annotations["left_events"]:
            text_lines.append("CURRENT FRAME = LEFT EVENT")
        if frame_idx in self.annotations["right_events"]:
            text_lines.append("CURRENT FRAME = RIGHT EVENT")

        y = 20
        for line in text_lines:
            draw.text((20, y), line, fill=(255, 0, 0))
            y += 25

        return img

    def show_frame(self, index):
        self.current_index = index
        img_path = self.image_files[index]
        filename = os.path.basename(img_path)

        img = Image.open(img_path).convert("RGB")
        img = self.resize_image(img)
        img = self.draw_overlay(img, index, filename)

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img)

        self.info_label.config(
            text=(
                f"Frame {index + 1}/{len(self.image_files)} | "
                f"Index={index} | "
                f"Left events={self.annotations['left_events']} | "
                f"Right events={self.annotations['right_events']}"
            )
        )

        if self.slider.get() != index:
            self.slider.set(index)

    def on_slider_move(self, value):
        self.show_frame(int(value))

    def prev_frame(self):
        if self.current_index > 0:
            self.show_frame(self.current_index - 1)

    def next_frame(self):
        if self.current_index < len(self.image_files) - 1:
            self.show_frame(self.current_index + 1)

    def jump_to_frame(self):
        try:
            idx = int(self.jump_entry.get().strip())
            if 0 <= idx < len(self.image_files):
                self.show_frame(idx)
            else:
                messagebox.showerror("Error", f"Frame index must be in [0, {len(self.image_files)-1}]")
        except:
            messagebox.showerror("Error", "Please enter a valid integer frame index")

    def mark_left_event(self):
        idx = self.current_index
        if idx not in self.annotations["left_events"]:
            self.annotations["left_events"].append(idx)
            self.annotations["left_events"] = sorted(self.annotations["left_events"])
        self.show_frame(idx)

    def mark_right_event(self):
        idx = self.current_index
        if idx not in self.annotations["right_events"]:
            self.annotations["right_events"].append(idx)
            self.annotations["right_events"] = sorted(self.annotations["right_events"])
        self.show_frame(idx)

    def delete_nearest(self, arr, target):
        if not arr:
            return arr, False
        nearest = min(arr, key=lambda x: abs(x - target))
        arr.remove(nearest)
        return arr, True

    def delete_nearest_left(self):
        self.annotations["left_events"], ok = self.delete_nearest(
            self.annotations["left_events"], self.current_index
        )
        self.show_frame(self.current_index)

    def delete_nearest_right(self):
        self.annotations["right_events"], ok = self.delete_nearest(
            self.annotations["right_events"], self.current_index
        )
        self.show_frame(self.current_index)

    def clear_events(self):
        has_events = bool(self.annotations["left_events"] or self.annotations["right_events"])
        if not has_events:
            self.show_frame(self.current_index)
            return

        ok = messagebox.askyesno(
            "Clear events",
            "Clear all LEFT and RIGHT events in memory?\n\nThis will not update the JSON file until you save."
        )
        if not ok:
            return

        self.annotations["left_events"] = []
        self.annotations["right_events"] = []
        self.show_frame(self.current_index)

    def build_stride_segments(self, events):
        events = sorted(events)
        segments = []
        for i in range(len(events) - 1):
            segments.append({
                "start_frame": events[i],
                "end_frame": events[i + 1],
                "duration_frames": events[i + 1] - events[i]
            })
        return segments

    def save_annotations(self):
        out = {
            "left_events": sorted(self.annotations["left_events"]),
            "right_events": sorted(self.annotations["right_events"]),
            "left_stride_segments": self.build_stride_segments(self.annotations["left_events"]),
            "right_stride_segments": self.build_stride_segments(self.annotations["right_events"])
        }

        with open(self.save_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print(f"Saved annotations to: {self.save_json_path}")
        messagebox.showinfo("Saved", f"Saved to:\n{self.save_json_path}")

    def export_stride_segments(self):
        self.save_annotations()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select frame folder")
    root.deiconify()

    if folder:
        app = StrideAnnotator(root, folder_path=folder)
        root.mainloop()
    else:
        print("No folder selected.")
