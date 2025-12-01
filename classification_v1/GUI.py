"""
Malaria Species Identification System - GUI Interface
"""


import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import csv
from datetime import datetime
from malaria_predict import malaria_predict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    print("Warning: tkinterdnd2 is not installed. Drag-and-drop functionality will be unavailable.")
    print("Attempt to install the command: pip install tkinterdnd2")

# Set Appearance Mode and Default Color Theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class MalariaClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BIA Group1 - Malaria Species Identification System")
        self.root.geometry("1200x800")
        self.root.resizable(False, False)

        
        # Images Storage
        self.images = []  # Store the path of uploaded images and results
        self.status = "initial"  # initial, uploaded, running, completed
        
        # Supported image formats
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')
        
        # Status Messages
        self.status_messages = {
            "initial": "Drag and drop files or click the Upload button to upload",
            "uploaded": "Click the Clear button to clear the cache or continue uploading files",
            "running": "Running...",
            "completed": "Click the Download button to download the CSV results file"
        }
        
        # Types of Malaria
        self.malaria_species = [
            "falciparum",
            "ovale",
            "vivax",
            "Uninfected"
        ]
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create User Interface"""
        # Main Interface
        main_container = ctk.CTkFrame(self.root, fg_color="#EEF2FF")
        main_container.pack(fill="both", expand=True)
        
        # Top area: Title (left) and button (right)
        top_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        top_frame.pack(fill="x", pady=(0, 15))
        
        # Left header area
        title_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        title_frame.pack(side="left", anchor="w", padx=(20, 0), pady=(15, 0))
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="BIA Group1",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#312E81"
        )
        title_label.pack(anchor="w")
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Malaria Species Identification",
            font=ctk.CTkFont(size=14),
            text_color="#4C51BF"
        )
        subtitle_label.pack(anchor="w")
        
        # Right button area
        button_container = ctk.CTkFrame(top_frame, fg_color="transparent")
        button_container.pack(side="right", anchor="e", padx=(0, 20), pady=(15, 0))
        
        # Four buttons (lined up in a row)
        buttons_grid = ctk.CTkFrame(button_container, fg_color="transparent")
        buttons_grid.pack()
        
        # ==================== Four buttons (lined up in a row) ====================
        
        # 1. Upload button
        self.upload_btn = ctk.CTkButton(
            buttons_grid,
            text="Upload",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#3B82F6",
            hover_color="#2563EB",
            text_color="white",
            corner_radius=15,
            height=45,
            width=120,
            command=self.upload_files
        )
        self.upload_btn.grid(row=0, column=0, padx=5)
        
        # 2. Clear button
        self.clear_btn = ctk.CTkButton(
            buttons_grid,
            text="Clear",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#D1D5DB",
            hover_color="#D1D5DB",
            text_color="white",
            corner_radius=15,
            height=45,
            width=120,
            command=self.clear_cache,
            state="disabled"
        )
        self.clear_btn.grid(row=0, column=1, padx=5)
        
        # 3. Run button
        self.run_btn = ctk.CTkButton(
            buttons_grid,
            text="Run",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#D1D5DB",
            hover_color="#D1D5DB",
            text_color="white",
            corner_radius=15,
            height=45,
            width=120,
            command=self.run_classification,
            state="disabled"
        )
        self.run_btn.grid(row=0, column=2, padx=5)
        
        # 4. Download button
        self.download_btn = ctk.CTkButton(
            buttons_grid,
            text="Download",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#D1D5DB",
            hover_color="#D1D5DB",
            text_color="white",
            corner_radius=15,
            height=45,
            width=120,
            command=self.download_results,
            state="disabled"
        )
        self.download_btn.grid(row=0, column=3, padx=5)
        
        # ==================== Status Indicator Box ====================
        self.status_frame = ctk.CTkFrame(
            main_container,
            fg_color="#EEF2FF",
            corner_radius=15,
            border_width=1,
            border_color="#C7D2FE"
        )
        self.status_frame.pack(fill="x", padx=(15, 15), pady=(0, 15))
        
        status_inner = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        status_inner.pack(padx=15, pady=12)
        
        # Status Indicator Point
        self.status_indicator = ctk.CTkLabel(
            status_inner,
            text="●",
            font=ctk.CTkFont(size=20),
            text_color="#9CA3AF"
        )
        self.status_indicator.pack(side="left", padx=(0, 10))
        
        # Status Text
        self.status_label = ctk.CTkLabel(
            status_inner,
            text=self.status_messages["initial"],
            font=ctk.CTkFont(size=14),
            text_color="#312E81"
        )
        self.status_label.pack(side="left")
        
        # Enable full-window drag functionality (if available)
        if DND_AVAILABLE:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)
        
        # Image Display Area (with scrolling)
        self.image_container_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        self.image_container_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Create a scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.image_container_frame,
            fg_color="white",
            corner_radius=20
        )
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        parent_canvas = self.scrollable_frame._parent_canvas  # internal canvas used by CTkScrollableFrame

        bg_dir = Path("./background")
        bg_files = sorted([p for p in bg_dir.glob("background*.png")])

        self.bg_tiles = []
        TARGET_W, TARGET_H = 320, 200

        for p in bg_files:
            try:
                img = Image.open(p).convert("RGBA")
                img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)   # unify size
                img.putalpha(40)   # semi-transparent
                self.bg_tiles.append(ImageTk.PhotoImage(img))
            except:
                pass

        parent_canvas = self.scrollable_frame._parent_canvas

        def draw_bg(event=None):
            if not self.bg_tiles:
                return

            parent_canvas.delete("bg")

            canvas_w = parent_canvas.winfo_width()
            canvas_h = parent_canvas.winfo_height()
            bbox = parent_canvas.bbox("all")
            region_h = max(canvas_h, bbox[3]) if bbox else canvas_h

            tile_w = TARGET_W
            tile_h = TARGET_H
            tile_index = 0

            x_positions = [-80, -320 -160, -240]

            row = 0
            y = 0
            while y < region_h:
                start_idx = row % len(x_positions)
                x_start = x_positions[start_idx]

                placed = 0
                x = x_start
                while placed < 5 and x < canvas_w + tile_w:
                    img_tile = self.bg_tiles[tile_index % len(self.bg_tiles)]
                    parent_canvas.create_image(
                        x, y, anchor="nw", image=img_tile, tags=("bg",)
                    )
                    tile_index += 1
                    placed += 1
                    x += tile_w

                y += tile_h
                row += 1

            parent_canvas.tag_lower("bg")


        self.draw_bg = draw_bg
        parent_canvas.bind("<Configure>", draw_bg)
        self.scrollable_frame.bind("<Configure>", draw_bg)
        self.root.after(50, draw_bg)

    def on_drop(self, event):
        """Handle dragged-and-dropped files"""
        files = self.root.tk.splitlist(event.data)
        self.add_files(files)
        
    def upload_files(self):
        """Upload button click event"""
        files = filedialog.askopenfilenames(
            title="Select image file",
            filetypes=[
                ("image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if files:
            self.add_files(files)
            
    def add_files(self, files):
        """Add files to the list"""
        # If the status is completed, clear the previous results
        if self.status == "completed":
            self.clear_cache()
        
        new_images = []
        invalid_files = []
        
        for file_path in files:
            if file_path.lower().endswith(self.supported_formats):
                new_images.append({
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'result': None
                })
            else:
                invalid_files.append(os.path.basename(file_path))
        
        # Update images and display
        if new_images:
            self.images.extend(new_images)
            self.display_images()
            self.update_buttons()
        
        # Display status message based on results
        if invalid_files and new_images:
            # Both valid and invalid files
            supported_formats_str = ", ".join([f.upper().replace('.', '') for f in self.supported_formats])
            if len(invalid_files) == 1:
                error_msg = f"✓ {len(new_images)} file(s) uploaded. ✗ 1 file skipped (unsupported format). Supported: {supported_formats_str}"
            else:
                error_msg = f"✓ {len(new_images)} file(s) uploaded. ✗ {len(invalid_files)} files skipped (unsupported formats). Supported: {supported_formats_str}"
            
            self.status_label.configure(text=error_msg)
            self.status_indicator.configure(text_color="#EAB308")  # Yellow/orange color for warning
            self.status = "uploaded"
        elif invalid_files:
            # Only invalid files
            supported_formats_str = ", ".join([f.upper().replace('.', '') for f in self.supported_formats])
            if len(invalid_files) == 1:
                error_msg = f"File format not supported: {invalid_files[0]}. Supported formats: {supported_formats_str}"
            else:
                error_msg = f"{len(invalid_files)} files with unsupported formats. Supported formats: {supported_formats_str}"
            
            self.status_label.configure(text=error_msg)
            self.status_indicator.configure(text_color="#DC2626")  # Red color for error
        elif new_images:
            # Only valid files
            self.update_status("uploaded")
        # else: No files were added and no invalid files (empty selection)
            
    def display_images(self):
        """Display uploaded images"""
        self.scrollable_frame.update_idletasks()
        # Clear existing display
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ctk.CTkCanvas):
                continue
            widget.destroy()

        if not self.images:
            if hasattr(self, "draw_bg"):
                self.root.after(50, self.draw_bg)
            return
        
        # title
        title = ctk.CTkLabel(
            self.scrollable_frame,
            text=f"Image uploaded ({len(self.images)})",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#312E81"
        )
        title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Create a grid layout
        grid_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # 4 cols
        cols = 4
        for i in range(cols):
            grid_frame.grid_columnconfigure(i, weight=0)
        
        # Show every Images
        for idx, img_data in enumerate(self.images):
            row = idx // cols
            col = idx % cols
            
            card_width = 300

            card = ctk.CTkFrame(
                grid_frame,
                fg_color="white",
                corner_radius=15,
                border_width=1,
                border_color="#E5E7EB",
                width=card_width,
                height=card_width
            )
            card.grid(row=row, column=col, padx=5, pady=5, sticky="e")

            img = Image.open(img_data['path']).convert("RGB")
            img = img.resize((int(card_width * 0.8), int(card_width * 0.8)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            
            img_label = ctk.CTkLabel(card, image=photo, text="")
            img_label.image = photo
            img_label.pack(padx=10, pady=(10, 5))
        
            # Show Filename
            filename_label = ctk.CTkLabel(
                card,
                text=img_data['filename'][:20] + "..." if len(img_data['filename']) > 20 else img_data['filename'],
                font=ctk.CTkFont(size=12),
                text_color="#374151"
            )
            filename_label.pack(padx=10, pady=(5, 5))
            
            # Result
            if img_data['result']:
                result_frame = ctk.CTkFrame(card, fg_color="transparent")
                result_frame.pack(padx=10, pady=(0, 10))
                
                species_label = ctk.CTkLabel(
                    result_frame,
                    text=img_data['result']['species'],
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#4C51BF"
                )
                species_label.pack()
                
                confidence_label = ctk.CTkLabel(
                    result_frame,
                    text=f"Confidence: {img_data['result']['confidence']*100:.2f}%",
                    font=ctk.CTkFont(size=10),
                    text_color="#6B7280"
                )
                confidence_label.pack()

        self.scrollable_frame.update_idletasks()
        self.scrollable_frame._parent_canvas.configure(scrollregion=self.scrollable_frame._parent_canvas.bbox("all"))

    
    def clear_cache(self):
        """Clear button click event"""
        # 销毁当前窗口下所有组件
        for widget in self.root.winfo_children():
            widget.destroy()

        # 重新运行 __init__ 初始化整个界面
        self.__init__(self.root)

    def run_classification(self):
        """Run Button Click Event - Run Category"""
        if not self.images:
            return
        
        # Disable all buttons
        self.update_status("running")
        self.update_buttons()
        self.run_btn.configure(text="⏳ Running...")
        self.root.update()
        
        model_path = Path("./model.ckpt")
        image_paths = [Path(img['path']) for img in self.images]
        df = malaria_predict(model_path, image_paths)
        
        for img_data, (_, row) in zip(self.images, df.iterrows()):
                img_data['result'] = {
                    'species': row['predicted_class'],
                    'confidence': float(row['confidence'])
                }

        self.update_status("completed")
        self.run_btn.configure(text="Run")
        self.display_images()
        self.update_buttons()
        
    def download_results(self):
        """Download Button Click Event - Download CSV Results"""
        if not self.images or not self.images[0]['result']:
            return
        
        # 选择保存位置
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"malaria_classification_results_{timestamp}.csv"
        
        file_path = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Filename', 'Species', 'Confidence'])
                    
                    for img_data in self.images:
                        if img_data['result']:
                            writer.writerow([
                                img_data['filename'],
                                img_data['result']['species'],
                                f"{img_data['result']['confidence']:.4f}"
                            ])
                
                # Display download completion information
                self.status_label.configure(
                    text=f"Download complete! File saved to: {file_path}"
                )
                self.status_indicator.configure(text_color="#16A34A")  # 绿色
                
            except Exception as e:
                # Display error message
                self.status_label.configure(
                    text=f"Save failed: {str(e)}"
                )
                self.status_indicator.configure(text_color="#DC2626")  # 红色
    
    def update_status(self, new_status):
        """Update Status"""
        self.status = new_status
        self.status_label.configure(text=self.status_messages[new_status])
        
        # Update status indicator color
        colors = {
            "initial": "#9CA3AF",
            "uploaded": "#3B82F6",
            "running": "#EAB308",
            "completed": "#16A34A"
        }
        self.status_indicator.configure(text_color=colors[new_status])
        
    def update_buttons(self):
        """Update button status"""
        has_images = len(self.images) > 0
        
        # Clear button
        if has_images:
            self.clear_btn.configure(state="normal")
            self.clear_btn.configure(fg_color="#EF4444", hover_color="#DC2626")
        else:
            self.clear_btn.configure(state="disabled")
            self.clear_btn.configure(fg_color="#D1D5DB", hover_color="#D1D5DB")
        
        # Run button
        if has_images and self.status != "running":
            self.run_btn.configure(state="normal")
            self.run_btn.configure(fg_color="#F59E0B", hover_color="#D97706")
        else:
            self.run_btn.configure(state="disabled")
            self.run_btn.configure(fg_color="#D1D5DB", hover_color="#D1D5DB")
        
        # Download button
        if self.status == "completed":
            self.download_btn.configure(state="normal")
            self.download_btn.configure(fg_color="#10B981", hover_color="#059669")
        else:
            self.download_btn.configure(state="disabled")
            self.download_btn.configure(fg_color="#D1D5DB", hover_color="#D1D5DB")


def main():
    """Main Function"""
    # If drag-and-drop is supported, use TkinterDnD.Tk; 
    # otherwise, use the standard CTk.
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = ctk.CTk()



    app = MalariaClassificationApp(root)
    
    root.mainloop()


if __name__ == "__main__":
    main()