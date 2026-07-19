import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import os
import pickle
import numpy as np

class HousePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("California House Price Predictor")
        self.root.geometry("550x650")
        self.root.resizable(False, False)
        
        # Set modern style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Define color scheme
        self.bg_color = "#f4f6f9"
        self.card_color = "#ffffff"
        self.primary_color = "#1a73e8"  # Google blue
        self.text_color = "#202124"
        self.accent_color = "#34a853"   # Green for price
        
        self.root.configure(bg=self.bg_color)
        
        # Load model
        self.model = None
        self.load_model()
        
        self.create_widgets()
        
    def load_model(self):
        model_path = "model.pkl"
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model pickle file '{model_path}' not found in the current directory.\nPlease run the model training script first to generate it.")
            self.root.destroy()
            return
            
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the model pickle file:\n{str(e)}")
            self.root.destroy()
            
    def create_widgets(self):
        # Header block
        header_frame = tk.Frame(self.root, bg=self.primary_color, height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame, 
            text="California House Price Predictor", 
            font=("Helvetica", 18, "bold"), 
            fg="white", 
            bg=self.primary_color
        )
        header_label.pack(pady=22)
        
        # Container frame
        container = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        container.pack(fill='both', expand=True)
        
        # Instruction
        instruction = tk.Label(
            container, 
            text="Enter features below to predict the median house value:",
            font=("Helvetica", 10, "italic"),
            fg="#5f6368",
            bg=self.bg_color
        )
        instruction.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))
        
        # Features input form
        features_info = [
            ("MedInc", "Median Income (in $10,000s):", "3.87", "Typical range: 0.5 - 15.0"),
            ("HouseAge", "Median House Age (years):", "28.6", "Range: 1 - 52"),
            ("AveRooms", "Average Rooms per Home:", "5.43", "Typical range: 3 - 8"),
            ("AveBedrms", "Average Bedrooms per Home:", "1.10", "Typical range: 0.8 - 1.5"),
            ("Population", "Block Population:", "1425.5", "Range: 3 - 35,000"),
            ("AveOccup", "Average Household Occupancy:", "3.07", "Typical range: 1.5 - 6"),
            ("Latitude", "Latitude (degrees):", "35.63", "Range: 32.5 - 41.9"),
            ("Longitude", "Longitude (degrees):", "-119.57", "Range: -124.3 - -114.3")
        ]
        
        self.entries = {}
        
        for i, (name, label_text, default, helper) in enumerate(features_info):
            # Label
            lbl = tk.Label(
                container, 
                text=label_text, 
                font=("Helvetica", 10, "bold"), 
                fg=self.text_color, 
                bg=self.bg_color
            )
            lbl.grid(row=i+1, column=0, sticky="w", pady=6)
            
            # Entry frame for alignment and styling
            entry_frame = tk.Frame(container, bg=self.bg_color)
            entry_frame.grid(row=i+1, column=1, sticky="ew", pady=6)
            
            entry = ttk.Entry(entry_frame, font=("Helvetica", 10), width=12)
            entry.insert(0, default)
            entry.pack(side="left")
            self.entries[name] = entry
            
            # Helper text
            help_lbl = tk.Label(
                entry_frame,
                text=f"  ({helper})",
                font=("Helvetica", 9),
                fg="#80868b",
                bg=self.bg_color
            )
            help_lbl.pack(side="left")
            
        container.columnconfigure(1, weight=1)
        
        # Buttons Frame
        btn_frame = tk.Frame(container, bg=self.bg_color, pady=15)
        btn_frame.grid(row=len(features_info)+1, column=0, columnspan=2)
        
        self.style.configure(
            "Primary.TButton", 
            font=("Helvetica", 11, "bold"), 
            foreground="white", 
            background=self.primary_color
        )
        self.style.map("Primary.TButton",
            background=[('active', '#1557b0'), ('pressed', '#1557b0')]
        )
        
        self.style.configure("Secondary.TButton", font=("Helvetica", 11))
        
        predict_btn = ttk.Button(
            btn_frame, 
            text="Predict Price", 
            style="Primary.TButton", 
            command=self.predict_price,
            width=15
        )
        predict_btn.pack(side="left", padx=10)
        
        reset_btn = ttk.Button(
            btn_frame, 
            text="Reset Defaults", 
            style="Secondary.TButton", 
            command=self.reset_defaults,
            width=15
        )
        reset_btn.pack(side="left", padx=10)
        
        # Result Card Block
        result_card = tk.Frame(
            container, 
            bg=self.card_color, 
            bd=1, 
            relief="groove", 
            padx=15, 
            pady=15
        )
        result_card.grid(row=len(features_info)+2, column=0, columnspan=2, sticky="ew", pady=10)
        
        result_title = tk.Label(
            result_card, 
            text="PREDICTED MEDIAN HOUSE VALUE", 
            font=("Helvetica", 9, "bold"), 
            fg="#5f6368", 
            bg=self.card_color
        )
        result_title.pack(anchor="center")
        
        self.result_val = tk.Label(
            result_card, 
            text="$0.00", 
            font=("Helvetica", 22, "bold"), 
            fg=self.accent_color, 
            bg=self.card_color
        )
        self.result_val.pack(anchor="center", pady=(5, 0))
        
    def predict_price(self):
        inputs = []
        for name, entry in self.entries.items():
            try:
                val = float(entry.get())
                inputs.append(val)
            except ValueError:
                messagebox.showerror("Input Error", f"Please enter a valid numeric value for {name}.")
                return
                
        # Predict using model
        input_array = np.array([inputs])
        predicted_val = self.model.predict(input_array)[0]
        
        # Convert predicted value (in units of $100,000) to actual dollars
        # Ensure value is not negative (linear regression can sometimes output negative for extreme values)
        actual_dollars = max(0.0, predicted_val * 100000.0)
        
        # Format output
        formatted_price = f"${actual_dollars:,.2f}"
        self.result_val.config(text=formatted_price)
        
    def reset_defaults(self):
        features_defaults = {
            "MedInc": "3.87",
            "HouseAge": "28.6",
            "AveRooms": "5.43",
            "AveBedrms": "1.10",
            "Population": "1425.5",
            "AveOccup": "3.07",
            "Latitude": "35.63",
            "Longitude": "-119.57"
        }
        for name, val in features_defaults.items():
            self.entries[name].delete(0, tk.END)
            self.entries[name].insert(0, val)
            
        self.result_val.config(text="$0.00")

if __name__ == "__main__":
    root = tk.Tk()
    app = HousePredictorApp(root)
    root.mainloop()
