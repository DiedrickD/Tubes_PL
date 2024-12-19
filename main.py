import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from stunting_tips import STUNTING_TIPS # Import stunting tips in bahasa

# Load the saved model and scaler
model = joblib.load('stunting_model.joblib')
scaler = joblib.load('stunting_scaler.joblib')

class StuntingPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Prediksi Stunting")
        self.root.geometry("800x500")
        self.root.configure(bg="#f0f4f8")

        # Custom style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TLabel', background="#f0f4f8", font=("Segoe UI", 12))
        self.style.configure('TEntry', font=("Segoe UI", 12))
        self.style.configure('TButton', font=("Segoe UI", 12, "bold"))
        self.style.map('TButton',
                       foreground=[('active', 'white')],
                       background=[('active', '#2c3e50')]
                       )

        # Status mapping
        self.status_map = {0: 'Normal', 1: 'Severely Stunted', 2: 'Stunted', 3: 'Tinggi'}

        self.create_ui()

    def create_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_label = tk.Label(header_frame,
                                text="Sistem Assessment Stunting Anak",
                                font=("Segoe UI", 22, "bold"),
                                fg="white",
                                bg="#2c3e50"
                                )
        header_label.pack(pady=20)

        # Main content frame
        content_frame = tk.Frame(self.root, bg="#f0f4f8", padx=30, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Card-like container
        card_frame = tk.Frame(content_frame, bg="white", bd=0, relief=tk.RAISED, highlightthickness=1,
                              highlightbackground="#e0e0e0")
        card_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        card_frame.grid_columnconfigure(1, weight=1)

        # Input Fields
        input_fields = [
            ("Umur (bulan)", "age"),
            ("Tinggi Badan (cm)", "height")
        ]

        for i, (label_text, entry_name) in enumerate(input_fields):
            label = ttk.Label(card_frame, text=label_text + ":", background="white")
            label.grid(row=i, column=0, sticky=tk.E, padx=10, pady=10)

            entry = ttk.Entry(card_frame, width=30)
            entry.grid(row=i, column=1, padx=10, pady=10, sticky=tk.EW)
            setattr(self, f"{entry_name}_entry", entry)

        # Gender input
        gender_label = ttk.Label(card_frame, text="Jenis Kelamin:", background="white")
        gender_label.grid(row=2, column=0, sticky=tk.E, padx=10, pady=10)

        self.gender_var = tk.StringVar(value="Laki-laki")
        gender_frame = tk.Frame(card_frame, bg="white")
        gender_frame.grid(row=2, column=1, pady=10, sticky=tk.W)

        gender_male = ttk.Radiobutton(gender_frame, text="Laki-laki", variable=self.gender_var, value="Laki-laki")
        gender_female = ttk.Radiobutton(gender_frame, text="Perempuan", variable=self.gender_var, value="Perempuan")
        gender_male.grid(row=0, column=0, padx=5)
        gender_female.grid(row=0, column=1, padx=5)

        # Buttons
        button_frame = tk.Frame(card_frame, bg="white")
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)

        predict_button = ttk.Button(button_frame, text="Prediksi", command=self.predict_stunting, width=20)
        predict_button.grid(row=0, column=0, padx=10)

        clear_button = ttk.Button(button_frame, text="Hapus", command=self.clear_form, width=20)
        clear_button.grid(row=0, column=1, padx=10)

        # Status bar
        self.status_var = tk.StringVar(value="Siap melakukan penilaian status stunting")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bd=1, relief=tk.SUNKEN, anchor=tk.W,
                              font=("Segoe UI", 10), bg="#f0f4f8")
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def predict_stunting(self):
        try:
            # Get input values
            age = int(self.age_entry.get())
            gender = self.gender_var.get()
            height = float(self.height_entry.get())

            # Convert gender to numerical value
            gender_numeric = 0 if gender == 'Laki-laki' else 1

            # Prepare input data as a DataFrame
            input_data = pd.DataFrame([[age, gender_numeric, height]],
                                      columns=['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)'])

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Make prediction and get probabilities
            prediction = model.predict(input_data_scaled)
            prediction_probabilities = model.predict_proba(input_data_scaled)

            # Get the predicted status
            status = self.status_map[prediction[0]]

            # Get the confidence level (highest probability)
            confidence = np.max(prediction_probabilities)

            # Show result in a custom pop-up window
            self.show_custom_popup(status, confidence, age, height, gender_numeric)

            # Update status bar
            self.status_var.set(f"Prediksi: {status} dengan keyakinan {confidence * 99.9:.1f}%")

        except ValueError:
            self.show_error_popup("Silakan masukkan data dengan benar!")

    def show_custom_popup(self, status, confidence, age, height, gender_numeric):
        # Create a new window (popup)
        popup = tk.Toplevel(self.root)
        popup.title("Hasil Penilaian")
        popup.geometry("600x700")
        popup.configure(bg="#f0f4f8")
        popup.resizable(True, True)

        # Header
        header_label = tk.Label(popup, text="Hasil Assessment Stunting",
                                font=("Segoe UI", 18, "bold"),
                                bg="#2c3e50",
                                fg="white")
        header_label.pack(fill=tk.X, pady=10)

        # Result details
        result_frame = tk.Frame(popup, bg="#f0f4f8")
        result_frame.pack(pady=10, padx=20, fill=tk.X)

        details = [
            f"Status Gizi: {status}",
            f"Tingkat Keyakinan: {confidence*99.9}%",
        ]

        for detail in details:
            label = tk.Label(result_frame, text=detail,
                             font=("Segoe UI", 14),
                             bg="#f0f4f8")
            label.pack(anchor='w')

        # Tips section with category-specific recommendations
        tips_text = tk.Text(popup, height=10, width=50,
                            font=("Segoe UI", 11),
                            bg="white", wrap=tk.WORD,
                            bd=1, relief=tk.SUNKEN, padx=10, pady=10)

        tips_text.insert(tk.END, STUNTING_TIPS.get(status, "Tidak ada rekomendasi spesifik"))
        tips_text.config(state=tk.DISABLED)
        tips_text.pack(pady=10)

        # Plot prediction graph
        self.plot_prediction_graph(age, height, status, popup, gender_numeric)

        # Close button
        close_button = ttk.Button(popup, text="Tutup", command=popup.destroy)
        close_button.pack(pady=20)

    def plot_prediction_graph(self, age, height, status, popup, gender_numeric):
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import tkinter as tk

        # Determine dataset and title based on gender
        if gender_numeric == 0:
            dataset_path = 'boys_combined_cleaned.csv'
            title = 'Z-score Trendline for Boys'
        else:
            dataset_path = 'girls_combined_cleaned.csv'
            title = 'Z-score Trendline for Girls'

        # Load the dataset
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"Error: {dataset_path} not found.")
            return

        # Create the trendline plot
        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure size for Tkinter

        predicted_month = age  # Assuming 'age' represents the predicted month
        df_filtered = df[(df['Month'] >= predicted_month) & (df['Month'] <= predicted_month + 10)]

        # Plot the filtered data
        ax.plot(df_filtered['Month'], df_filtered['SD3neg'], label='-3 SD', color='red', linewidth=2)
        ax.plot(df_filtered['Month'], df_filtered['SD2neg'], label='-2 SD', color='orange', linewidth=2)
        ax.plot(df_filtered['Month'], df_filtered['SD1neg'], label='-1 SD', color='yellow', linewidth=2)
        ax.plot(df_filtered['Month'], df_filtered['SD0'], label='Median', color='green', linewidth=2)
        ax.plot(df_filtered['Month'], df_filtered['SD1'], label='+1 SD', color='yellow', linewidth=2)
        ax.plot(df_filtered['Month'], df_filtered['SD2'], label='+2 SD', color='orange', linewidth=2)
        ax.plot(df_filtered['Month'], df_filtered['SD3'], label='+3 SD', color='red', linewidth=2)

        plt.scatter(age, height, color='blue', zorder=5, label=f"Prediksi: {status}")

        # Customize the plot
        ax.set_xlabel('Age (months)', fontsize=10)
        ax.set_ylabel('Z-score', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(title='SD Categories', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout tightly
        fig.tight_layout()

        # Embed the plot in the Tkinter popup window
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=False, fill=tk.NONE, padx=10, pady=10)  # Center it with padding

        # Render the plot
        canvas.draw()

    def show_error_popup(self, message):
        # Error popup
        popup = tk.Toplevel(self.root)
        popup.title("Kesalahan Input")
        popup.geometry("300x150")
        popup.configure(bg="#f0f4f8")
        popup.resizable(False, False)

        error_label = tk.Label(popup, text=message,
                               font=("Segoe UI", 12),
                               fg="red",
                               bg="#f0f4f8")
        error_label.pack(pady=30)

        close_button = ttk.Button(popup, text="Tutup", command=popup.destroy)
        close_button.pack()

    def clear_form(self):
        # Clear input fields
        self.age_entry.delete(0, tk.END)
        self.height_entry.delete(0, tk.END)
        self.gender_var.set("Laki-laki")

        # Reset status bar
        self.status_var.set("Siap melakukan penilaian status stunting")

def main():
    root = tk.Tk()
    app = StuntingPredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()