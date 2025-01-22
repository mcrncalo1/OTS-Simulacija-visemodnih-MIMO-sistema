import tkinter as tk
from tkinter import ttk, Toplevel, Label, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy import signal
from numpy.linalg import svd, det, matrix_rank, inv
from typing import Tuple, List, Dict
from enum import Enum

class ToolTip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = Label(self.tooltip_window, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, padx=1, pady=1)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()

class QPSK_MIMO_GUI:
    SNR_RANGE = (0, 20)
    SNR_POINTS = 10
    FIBER_LENGTH_POINTS = 100
    RANDOM_SEED = 42

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("QPSK MIMO Simulacija")
        # master.attributes('-fullscreen', True)  # Start in full-screen

        # Simulation results frame (top right)
        self.results_frame = ttk.LabelFrame(master, text="Rezultati simulacije")
        self.results_frame.pack(padx=10, pady=5, anchor=tk.NE, side=tk.TOP)
        ToolTip(self.results_frame, "Prikazuje rezultate simulacije kao što su BER, SNR i kapacitet.")

        self.ber_label_text = tk.StringVar()
        self.ber_label_text.set("BER: N/A")
        self.ber_label = ttk.Label(self.results_frame, textvariable=self.ber_label_text)
        self.ber_label.pack(padx=5, pady=2)

        self.snr_result_label_text = tk.StringVar()
        self.snr_result_label_text.set("SNR (dB): N/A")
        self.snr_result_label = ttk.Label(self.results_frame, textvariable=self.snr_result_label_text)
        self.snr_result_label.pack(padx=5, pady=2)

        self.capacity_label_text = tk.StringVar()
        self.capacity_label_text.set("Kapacitet (bps/Hz): N/A")
        self.capacity_label = ttk.Label(self.results_frame, textvariable=self.capacity_label_text)
        self.capacity_label.pack(padx=5, pady=2)

        # Input parameters frame
        self.input_frame = ttk.LabelFrame(master, text="Parametri simulacije")
        self.input_frame.pack(padx=10, pady=10, fill=tk.X, anchor=tk.NW, side=tk.TOP)

        # Number of bits
        self.num_bits_label = ttk.Label(self.input_frame, text="Broj bita (100-1000):")
        self.num_bits_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_bits_entry = ttk.Entry(self.input_frame)
        self.num_bits_entry.insert(0, "500")
        self.num_bits_entry.grid(row=0, column=1, padx=5, pady=5)
        ToolTip(self.num_bits_label, "Ukupan broj generisanih bitova za simulaciju.")

        # SNR (dB)
        self.snr_label = ttk.Label(self.input_frame, text="SNR (dB) (0-30):")
        self.snr_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.snr_entry = ttk.Entry(self.input_frame)
        self.snr_entry.insert(0, "10")
        self.snr_entry.grid(row=1, column=1, padx=5, pady=5)
        ToolTip(self.snr_label, "Omjer signala i šuma u decibelima.")

        # Broj predajnih antena
        self.num_tx_ant_label = ttk.Label(self.input_frame, text="Broj predajnih antena (1-4):")
        self.num_tx_ant_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_tx_ant_entry = ttk.Entry(self.input_frame)
        self.num_tx_ant_entry.insert(0, "2")
        self.num_tx_ant_entry.grid(row=2, column=1, padx=5, pady=5)
        ToolTip(self.num_tx_ant_label, "Broj predajnih antena u MIMO sistemu.")
        self.num_tx_ant_entry.bind("<FocusOut>", self.update_channel_matrix_size)

        # Broj prijemnih antena
        self.num_rx_ant_label = ttk.Label(self.input_frame, text="Broj prijemnih antena (1-4):")
        self.num_rx_ant_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_rx_ant_entry = ttk.Entry(self.input_frame)
        self.num_rx_ant_entry.insert(0, "2")
        self.num_rx_ant_entry.grid(row=3, column=1, padx=5, pady=5)
        ToolTip(self.num_rx_ant_label, "Broj prijemnih antena u MIMO sistemu.")
        self.num_rx_ant_entry.bind("<FocusOut>", self.update_channel_matrix_size)

        # Broj modova
        self.num_modes_label = ttk.Label(self.input_frame, text="Broj modova (1-4):")
        self.num_modes_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_modes_entry = ttk.Entry(self.input_frame)
        self.num_modes_entry.insert(0, "2")
        self.num_modes_entry.grid(row=4, column=1, padx=5, pady=5)
        ToolTip(self.num_modes_label, "Broj prostornih modova u vlaknu.")
        self.num_modes_entry.bind("<FocusOut>", self.update_channel_matrix_size)

        # Kanalna matrica
        self.channel_label = ttk.Label(self.input_frame, text="Kanalna matrica (H):")
        self.channel_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.channel_entry = ttk.Entry(self.input_frame)
        self.channel_entry.insert(0, "[[1, 0.5], [0.5, 1]]")
        self.channel_entry.grid(row=5, column=1, padx=5, pady=5)
        #self.channel_entry.config(state='readonly')
        ToolTip(self.channel_label, "Matrica koja opisuje propagaciju signala između predajnih i prijemnih antena/modova.")

        # Duljina vlakna (km)
        self.fiber_length_label = ttk.Label(self.input_frame, text="Dužina vlakna (km) (1-1000):")
        self.fiber_length_label.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        self.fiber_length_entry = ttk.Entry(self.input_frame)
        self.fiber_length_entry.insert(0, "100")
        self.fiber_length_entry.grid(row=6, column=1, padx=5, pady=5)
        ToolTip(self.fiber_length_label, "Dužina optičkog vlakna u kilometrima.")

        # Koeficijent slabljenja (dB/km)
        self.attenuation_label = ttk.Label(self.input_frame, text="Koef. slabljenja (dB/km) (0.1-1):")
        self.attenuation_label.grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)
        self.attenuation_entry = ttk.Entry(self.input_frame)
        self.attenuation_entry.insert(0, "0.2")
        self.attenuation_entry.grid(row=7, column=1, padx=5, pady=5)
        ToolTip(self.attenuation_label, "Koeficijent slabljenja signala po kilometru vlakna.")

        self.explain_button = ttk.Button(master, text="Objasni koncept", command=self.explain_concept)
        self.explain_button.pack(pady=5, side=tk.RIGHT, padx=10, anchor=tk.NE)

        self.help_button = ttk.Button(master, text="Pomoć", command=self.show_help)
        self.help_button.pack(pady=5, side=tk.RIGHT, padx=10, anchor=tk.NE)

        self.simulate_button = ttk.Button(master, text="Simuliraj", command=self.simulate)
        self.simulate_button.pack(pady=5, side=tk.LEFT, padx=10, anchor=tk.NW)

        self.reset_button = ttk.Button(master, text="Resetuj", command=self.reset_simulation)
        self.reset_button.pack(pady=5, side=tk.LEFT, padx=10, anchor=tk.NW)

        # Notebook for tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10, side=tk.TOP)

        # Tab 1: Transmitted Signal
        self.tx_signal_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tx_signal_tab, text="Odašiljani signal")
        self.tx_signal_figure, self.tx_signal_ax = plt.subplots()
        self.tx_signal_canvas = FigureCanvasTkAgg(self.tx_signal_figure, master=self.tx_signal_tab)
        self.tx_signal_canvas_widget = self.tx_signal_canvas.get_tk_widget()
        self.tx_signal_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.tx_signal_figure.tight_layout(pad=3.0)

        # Tab 2: Constellation Diagram
        self.constellation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.constellation_tab, text="Konstelacija")
        self.constellation_figure, self.constellation_ax = plt.subplots()
        self.constellation_canvas = FigureCanvasTkAgg(self.constellation_figure, master=self.constellation_tab)
        self.constellation_canvas_widget = self.constellation_canvas.get_tk_widget()
        self.constellation_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.constellation_figure.tight_layout(pad=3.0)

        # Tab 3: Channel Matrix
        self.channel_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.channel_tab, text="Kanalna matrica")
        self.channel_figure, self.channel_ax = plt.subplots()
        self.channel_canvas = FigureCanvasTkAgg(self.channel_figure, master=self.channel_tab)
        self.channel_canvas_widget = self.channel_canvas.get_tk_widget()
        self.channel_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.channel_figure.tight_layout(pad=3.0)

        # Tab 4: Eye Diagram
        self.eye_diagram_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.eye_diagram_tab, text="Eye Dijagram")
        self.eye_diagram_figure, self.eye_diagram_ax = plt.subplots()
        self.eye_diagram_canvas = FigureCanvasTkAgg(self.eye_diagram_figure, master=self.eye_diagram_tab)
        self.eye_diagram_canvas_widget = self.eye_diagram_canvas.get_tk_widget()
        self.eye_diagram_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.eye_diagram_figure.tight_layout(pad=3.0)

        # Tab 5: Utjecaj šuma
        self.noise_impact_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.noise_impact_tab, text="Utjecaj šuma na signal")
        self.noise_impact_figure, self.noise_impact_ax = plt.subplots()
        self.noise_impact_canvas = FigureCanvasTkAgg(self.noise_impact_figure, master=self.noise_impact_tab)
        self.noise_impact_canvas_widget = self.noise_impact_canvas.get_tk_widget()
        self.noise_impact_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.noise_impact_figure.tight_layout(pad=3.0)

        # Tab 6: SNR vs BER
        self.snr_ber_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.snr_ber_tab, text="SNR vs BER")
        self.snr_ber_figure, self.snr_ber_ax = plt.subplots()
        self.snr_ber_canvas = FigureCanvasTkAgg(self.snr_ber_figure, master=self.snr_ber_tab)
        self.snr_ber_canvas_widget = self.snr_ber_canvas.get_tk_widget()
        self.snr_ber_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.snr_ber_figure.tight_layout(pad=3.0)

        # Tab 7: SNR vs Kapacitet
        self.snr_capacity_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.snr_capacity_tab, text="SNR vs Kapacitet")
        self.snr_capacity_figure, self.snr_capacity_ax = plt.subplots()
        self.snr_capacity_canvas = FigureCanvasTkAgg(self.snr_capacity_figure, master=self.snr_capacity_tab)
        self.snr_capacity_canvas_widget = self.snr_capacity_canvas.get_tk_widget()
        self.snr_capacity_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.snr_capacity_figure.tight_layout(pad=3.0)

        # Tab 8: Detaljno vlakno
        self.detailed_fiber_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.detailed_fiber_tab, text="Detaljni prikaz vlakna")
        self.detailed_fiber_figure, self.detailed_fiber_ax = plt.subplots()
        self.detailed_fiber_canvas = FigureCanvasTkAgg(self.detailed_fiber_figure, master=self.detailed_fiber_tab)
        self.detailed_fiber_canvas_widget = self.detailed_fiber_canvas.get_tk_widget()
        self.detailed_fiber_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.detailed_fiber_figure.tight_layout(pad=3.0)

        self.channel_matrix_displayed = False
        self.fiber_propagation_ax = None
        self.fiber_propagation_canvas = None
        self.update_channel_matrix_size()

        # Define constants
        self.SNR_RANGE = (0, 20)
        self.SNR_POINTS = 10
        self.FIBER_LENGTH_POINTS = 100
        # np.random.seed(42) # Seed for reproducibility

    def show_help(self):
        help_text = """
        Ovaj program simulira QPSK MIMO sistem zasnovan na višemodnom optičkom vlaknu.

        **Parametri simulacije:**
        - **Broj bita:** Ukupan broj generisanih bitova za simulaciju.
        - **SNR (dB):** Omjer signala i šuma u decibelima.
        - **Broj predajnih antena:** Broj predajnih antena u MIMO sistemu.
        - **Broj prijemnih antena:** Broj prijemnih antena u MIMO sistemu.
        - **Broj modova:** Broj prostornih modova u vlaknu.
        - **Kanalna matrica (H):** Matrica koja opisuje propagaciju signala između predajnih i prijemnih antena/modova.
        - **Dužina vlakna (km):** Dužina optičkog vlakna u kilometrima.
        - **Koef. slabljenja (dB/km):** Koeficijent slabljenja signala po kilometru vlakna.

        **Rezultati simulacije:**
        - **BER:** Bit Error Rate - omjer broja pogrešno primljenih bitova i ukupnog broja poslanih bitova.
        - **SNR (dB):** Omjer signala i šuma na prijemu.
        - **Kapacitet (bps/Hz):** Maksimalna brzina prijenosa podataka po jedinici frekvencije.

        **Grafovi:**
        - **Odašiljani signal:** Prikazuje konstelaciju odašiljanih QPSK simbola.
        - **Konstelacija:** Prikazuje konstelaciju primljenih simbola nakon prolaska kroz kanal i dodavanja šuma.
        - **Kanalna matrica:** Prikazuje matricu kanala H.
        - **Eye Dijagram:** Prikazuje eye dijagram primljenog signala.
        - **Utjecaj šuma na signal:** Prikazuje utjecaj šuma na odašiljani i primljeni signal.
        - **SNR vs BER:** Prikazuje ovisnost BER o SNR.
        - **SNR vs Kapacitet:** Prikazuje ovisnost kapaciteta o SNR.
        - **Detaljni prikaz vlakna:** Prikazuje slabljenje signala duž vlakna.

        **Kako koristiti:**
        1. Unesite željene parametre simulacije.
        2. Kliknite "Simuliraj" za pokretanje simulacije.
        3. Pregledajte rezultate i grafove u odgovarajućim tabovima.
        4. Koristite "Resetuj" za brisanje svih grafova i rezultata.
        """
        help_window = tk.Toplevel(self.master)
        help_window.title("Pomoć")
        help_label = ttk.Label(help_window, text=help_text, wraplength=600, justify=tk.LEFT)
        help_label.pack(padx=10, pady=10)

    def explain_concept(self):
        explanation = """
        QPSK MIMO modulacija zasnovana na višemodnom optičkom vlaknu koristi se za slanje podataka kodiranjem u fazu signala (QPSK) i korištenjem više prostornih modova svjetlosti u vlaknu (MIMO).

        * **QPSK (Quadrature Phase-Shift Keying):**  Digitalna modulacijska tehnika gdje se dva bita podataka kodiraju u četiri moguće faze nosivog signala.

        * **MIMO (Multiple-Input Multiple-Output):** Komunikacijski sistem koji koristi više predajnih i prijemnih antena za povećanje propusnosti podataka i pouzdanosti. U kontekstu višemodnog vlakna, različiti prostorni modovi svjetlosti se tretiraju kao nezavisni kanali.

        * **Višemodno optičko vlakno (Multimode optical fiber):** Optičko vlakno koje može prenositi više svjetlosnih zraka (modova) istovremeno. Svaki mod se može koristiti kao nezavisni kanal u MIMO sistemu.

        **Uloga Python biblioteka:**

        * **NumPy:** Koristi se za matematičke operacije nad signalima i matricama kanala. Na primjer, za generisanje signala, manipulaciju matricama kanala i proračun performansi.

        * **Matplotlib:** Koristi se za vizualizaciju rezultata, kao što su konstelacijski dijagrami moduliranih signala, grafovi performansi sistema (npr. BER u odnosu na SNR) i prikaz prostornih modova.

        * **SciPy:** Koristi se za naprednije funkcije obrade signala, kao što su filtriranje, transformacije (npr. FFT za analizu frekvencijskog spektra) i potencijalno za modeliranje propagacije svjetlosti kroz vlakno (iako je to složenije i može zahtijevati specijalizirane biblioteke).

        Ova simulacija će demonstrirati osnovni QPSK modulator i prikazati konstelacijski dijagram. Za potpunu MIMO simulaciju sa višemodnim vlaknom, potrebno je modelirati propagaciju kroz vlakno i efekte miješanja modova.
        """
        explanation_window = tk.Toplevel(self.master)
        explanation_window.title("Objašnjenje koncepta")
        explanation_label = ttk.Label(explanation_window, text=explanation, wraplength=500)
        explanation_label.pack(padx=10, pady=10)

    def reset_simulation(self):
        # Clear all plots
        self.tx_signal_ax.clear()
        self.tx_signal_canvas.draw()
        self.constellation_ax.clear()
        self.constellation_canvas.draw()
        self.channel_ax.clear()
        self.channel_canvas.draw()
        self.snr_ber_ax.clear()
        self.snr_ber_canvas.draw()
        self.snr_capacity_ax.clear()
        self.snr_capacity_canvas.draw()
        self.noise_impact_ax.clear()
        self.noise_impact_canvas.draw()
        self.detailed_fiber_ax.clear()
        self.detailed_fiber_canvas.draw()
        self.eye_diagram_ax.clear()
        self.eye_diagram_canvas.draw()
        if self.fiber_propagation_ax:
            self.fiber_propagation_ax.clear()
            self.fiber_propagation_canvas.draw()

        # Reset result labels
        self.ber_label_text.set("BER: N/A")
        self.snr_result_label_text.set("SNR (dB): N/A")
        self.capacity_label_text.set("Kapacitet (bps/Hz): N/A")
        self.channel_matrix_displayed = False

    def update_channel_matrix_size(self, event=None):
        try:
            self.channel_entry.config(state='normal')
            num_modes = int(self.num_modes_entry.get())
            if not 1 <= num_modes <= 4:
                messagebox.showerror("Greška", "Broj modova mora biti između 1 i 4.")
                self.num_modes_entry.delete(0, tk.END)
                self.num_modes_entry.insert(0, "1")
                return
            num_tx_antennas = int(self.num_tx_ant_entry.get())
            if not 1 <= num_tx_antennas <= 4:
                messagebox.showerror("Greška", "Broj predajnih antena mora biti između 1 i 4.")
                self.num_tx_ant_entry.delete(0, tk.END)
                self.num_tx_ant_entry.insert(0, "1")
                return
            num_rx_antennas = int(self.num_rx_ant_entry.get())
            if not 1 <= num_rx_antennas <= 4:
                messagebox.showerror("Greška", "Broj prijemnih antena mora biti između 1 i 4.")
                self.num_rx_ant_entry.delete(0, tk.END)
                self.num_rx_ant_entry.insert(0, "1")
                return
            new_matrix_size = (num_rx_antennas * num_modes, num_tx_antennas * num_modes)
            default_matrix = np.eye(min(new_matrix_size))

            if new_matrix_size[0] > new_matrix_size[1]:
                padding = np.zeros((new_matrix_size[0] - new_matrix_size[1], new_matrix_size[1]))
                default_matrix = np.vstack((default_matrix, padding))
            elif new_matrix_size[1] > new_matrix_size[0]:
                padding = np.zeros((new_matrix_size[0], new_matrix_size[1] - new_matrix_size[0]))
                default_matrix = np.hstack((default_matrix, padding))

            self.channel_entry.delete(0, tk.END)
            self.channel_entry.insert(0, str(default_matrix.tolist()))
            self.channel_entry.config(state='readonly')
        except ValueError:
            messagebox.showerror("Greška", "Neispravan unos za broj modova.")
            self.num_modes_entry.delete(0, tk.END)
            self.num_modes_entry.insert(0, "1")

            self.channel_entry.config(state='readonly')

    def simulate(self):
        try:
            np.random.seed(42) # Seed for reproducibility
            # Dohvati parametre simulacije iz GUI
            try:
                num_bits = int(self.num_bits_entry.get())
            except ValueError:
                messagebox.showerror("Greška", "Broj bita mora biti cijeli broj.")
                return
            if not 100 <= num_bits <= 1000:
                messagebox.showerror("Greška", "Broj bita mora biti između 100 i 1000.")
                return

            try:
                snr_db = float(self.snr_entry.get())
            except ValueError:
                messagebox.showerror("Greška", "SNR mora biti broj.")
                return
            if not 0 <= snr_db <= 30:
                messagebox.showerror("Greška", "SNR mora biti između 0 i 30 dB.")
                return

            try:
                fiber_length = float(self.fiber_length_entry.get())
            except ValueError:
                messagebox.showerror("Greška", "Dužina vlakna mora biti broj.")
                return
            if not 1 <= fiber_length <= 1000:
                messagebox.showerror("Greška", "Dužina vlakna mora biti između 1 i 1000 km.")
                return

            try:
                attenuation = float(self.attenuation_entry.get())
            except ValueError:
                 messagebox.showerror("Greška", "Koeficijent slabljenja mora biti broj.")
                 return
            if not 0.1 <= attenuation <= 1:
                messagebox.showerror("Greška", "Koeficijent slabljenja mora biti između 0.1 i 1 dB/km.")
                return

            try:
                num_modes = int(self.num_modes_entry.get())
            except ValueError:
                messagebox.showerror("Greška", "Broj modova mora biti cijeli broj.")
                return
            if not 1 <= num_modes <= 4:
                messagebox.showerror("Greška", "Broj modova mora biti između 1 i 4.")
                return

            try:
                H_str = self.channel_entry.get()
                H = np.array(eval(H_str))
                num_tx_antennas = int(self.num_tx_ant_entry.get())
                if not 1 <= num_tx_antennas <= 4:
                    messagebox.showerror("Greška", "Broj predajnih antena mora biti između 1 i 4.")
                    return
                num_rx_antennas = int(self.num_rx_ant_entry.get())
                if not 1 <= num_rx_antennas <= 4:
                    messagebox.showerror("Greška", "Broj prijemnih antena mora biti između 1 i 4.")
                    return
                if H.shape[0] != num_rx_antennas * num_modes or H.shape[1] != num_tx_antennas * num_modes:
                    messagebox.showerror("Greška", "Dimenzije kanalne matrice ne odgovaraju broju antena i modova.")
                    return
            except:
                messagebox.showerror("Greška", "Neispravan format kanalne matrice.")
                return

            np.random.seed(self.RANDOM_SEED)
            bits = np.random.randint(0, 2, num_bits)

            # QPSK Modulacija
            qpsk_symbols, tx_bits = self._qpsk_modulate(bits)

            # MIMO dio
            tx_signals = np.tile(qpsk_symbols, (num_tx_antennas * num_modes, 1))

            # Prijem signala
            rx_signals = np.dot(H, tx_signals)

            # Dodavanje šuma (AWGN)
            noise = self._generate_noise(rx_signals, snr_db)
            received_symbols = rx_signals + noise

            # Demodulacija
            demodulated_bits = self._demodulate(received_symbols, self._get_qpsk_mapping())

            # Proračun BER
            ber_point = self._calculate_ber(tx_bits, demodulated_bits)
            if not np.isnan(ber_point):
                self.ber_label_text.set(f"BER: {ber_point:.4f}")
            else:
                self.ber_label_text.set("BER: N/A")

            self.snr_result_label_text.set(f"SNR (dB): {snr_db:.2f}")

            # Proračun kapaciteta (pojednostavljeno za AWGN kanal)
            capacity_point = self._calculate_capacity(H, snr_db, num_tx_antennas, num_rx_antennas, num_modes)
            if not np.isnan(capacity_point):
                self.capacity_label_text.set(f"Kapacitet (bps/Hz): {capacity_point:.2f}")
            else:
                self.capacity_label_text.set("Kapacitet (bps/Hz): N/A")

            # Prikaz odasiljanog signala (prva antena)
            self._plot_tx_signal(qpsk_symbols)

            # Prikaz konstelacije
            self._plot_constellation(received_symbols, self._get_qpsk_mapping(), num_rx_antennas, num_modes)

            # Prikaz kanalne matrice
            self._plot_channel_matrix(H, num_tx_antennas, num_rx_antennas, num_modes)

            # SNR vs BER plot
            snr_db_range = np.linspace(self.SNR_RANGE[0], self.SNR_RANGE[1], self.SNR_POINTS)
            ber_values = []
            for snr_db_val in snr_db_range:
                received_symbols_ber = self._generate_noise(rx_signals, snr_db_val)
                received_symbols_ber = rx_signals + received_symbols_ber
                demodulated_bits_ber = self._demodulate(received_symbols_ber, self._get_qpsk_mapping())
                ber = self._calculate_ber(tx_bits, demodulated_bits_ber)
                ber_values.append(ber)

            self._plot_snr_ber(snr_db_range, ber_values)

            # SNR vs Kapacitet plot
            capacity_values = []
            for snr_db_val in snr_db_range:
                capacity = self._calculate_capacity(H, snr_db_val, num_tx_antennas, num_rx_antennas, num_modes)
                capacity_values.append(capacity)

            self._plot_snr_capacity(snr_db_range, capacity_values)

            # Utjecaj šuma na signal
            self._plot_noise_impact(tx_signals, noise, received_symbols)

            # Detaljni prikaz vlakna
            self._plot_detailed_fiber(fiber_length, attenuation)

            # Eye Diagram
            self._plot_eye_diagram(received_symbols, num_tx_antennas, num_modes)

        except Exception as e:
            messagebox.showerror("Greška", f"Došlo je do neočekivane greške: {e}")
            return

    def _plot_noise_impact(self, tx_signals: np.ndarray, noise: np.ndarray, received_symbols: np.ndarray):
        """Plot the impact of noise on the signal."""
        self.noise_impact_ax.clear()
        if tx_signals.size > 0 and noise.size > 0:
            self.noise_impact_ax.plot(np.real(tx_signals[0, :]), label='Odašiljani signal (I komponenta)')
            self.noise_impact_ax.plot(np.imag(tx_signals[0, :]), label='Odašiljani signal (Q komponenta)')
            self.noise_impact_ax.plot(np.real(noise[0, :]), label='Šum (I komponenta)')
            self.noise_impact_ax.plot(np.imag(noise[0, :]), label='Šum (Q komponenta)')
            self.noise_impact_ax.plot(np.real(received_symbols[0, :]), label='Primljeni signal (I komponenta)')
            self.noise_impact_ax.plot(np.imag(received_symbols[0, :]), label='Primljeni signal (Q komponenta)')
            self.noise_impact_ax.set_xlabel('Vrijeme (uzorci)')
            self.noise_impact_ax.set_ylabel('Amplituda')
            self.noise_impact_ax.set_title('Utjecaj šuma na QPSK signal')
            self.noise_impact_ax.legend()
        self.noise_impact_canvas.draw()

    def _generate_bits(self, num_bits: int) -> np.ndarray:
        """Generate random bits."""
        return np.random.randint(0, 2, num_bits)

    def _qpsk_modulate(self, bits: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """QPSK modulate the bits."""
        qpsk_symbols = []
        tx_bits = []
        qpsk_mapping = self._get_qpsk_mapping()
        for i in range(0, len(bits), 2):
            if i + 1 < len(bits):
                tx_bits.extend([bits[i], bits[i+1]])
                symbol = qpsk_mapping.get((bits[i], bits[i+1]))
                if symbol is None:
                    raise ValueError("Invalid bit combination")
                qpsk_symbols.append(symbol)
        return np.array(qpsk_symbols), tx_bits

    def _generate_noise(self, signals: np.ndarray, snr_db: float) -> np.ndarray:
        """Generate AWGN noise."""
        snr_linear = 10**(snr_db / 10)
        signal_power = np.mean(np.abs(signals)**2)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(*signals.shape) + 1j * np.random.randn(*signals.shape))
        return noise

    def _get_qpsk_mapping(self) -> Dict[Tuple[int, int], complex]:
        """Return the QPSK mapping dictionary."""
        return {
            (0, 0): complex(1/np.sqrt(2), 1/np.sqrt(2)),  # 00 -> 1+j
            (0, 1): complex(-1/np.sqrt(2), 1/np.sqrt(2)), # 01 -> -1+j
            (1, 0): complex(1/np.sqrt(2), -1/np.sqrt(2)), # 10 -> 1-j
            (1, 1): complex(-1/np.sqrt(2), -1/np.sqrt(2))  # 11 -> -1-j
        }

    def _demodulate(self, received_symbols: np.ndarray, qpsk_mapping) -> List[int]:
        """Demodulate the received symbols."""
        inverse_mapping = {value: key for key, value in qpsk_mapping.items()}
        demodulated_bits = []
        for rx_signal in received_symbols.T:  # Iterate through received symbols
            for symbol in rx_signal:
                min_dist = float('inf')
                closest_symbol = None
                for ref_symbol in qpsk_mapping.values():
                    dist = np.abs(symbol - ref_symbol)**2
                    if dist < min_dist:
                        min_dist = dist
                        closest_symbol = ref_symbol
                if closest_symbol is not None:
                    demodulated_bits.extend(inverse_mapping[closest_symbol])
        return demodulated_bits

    def _calculate_ber(self, tx_bits: List[int], rx_bits: List[int]) -> float:
        """Calculate the Bit Error Rate."""
        if not tx_bits or not rx_bits:
            return np.nan
        min_length = min(len(tx_bits), len(rx_bits))
        return np.mean(np.array(tx_bits[:min_length]) != np.array(rx_bits[:min_length]))

    def _calculate_capacity(self, H: np.ndarray, snr_db: float, num_tx_antennas: int, num_rx_antennas: int, num_modes: int) -> float:
        """Calculate the channel capacity."""
        snr_linear = 10**(snr_db / 10)
        if num_rx_antennas * num_modes == 0 or num_tx_antennas * num_modes == 0:
            return np.nan
        if H.shape[0] != H.shape[1]:
            return np.nan
        return np.log2(det(np.eye(num_rx_antennas * num_modes) + (snr_linear / (num_tx_antennas * num_modes)) * np.dot(H, H.conj().T)))

    def _plot_tx_signal(self, qpsk_symbols: np.ndarray):
        """Plot the transmitted signal."""
        self.tx_signal_ax.clear()
        self.tx_signal_ax.plot(qpsk_symbols.real, qpsk_symbols.imag, 'o', label='Odašiljani simboli')
        self.tx_signal_ax.set_xlabel('In-phase komponenta')
        self.tx_signal_ax.set_ylabel('Quadrature komponenta')
        self.tx_signal_ax.set_title('Odašiljani signal')
        self.tx_signal_ax.grid(True)
        self.tx_signal_ax.axis('equal')
        self.tx_signal_ax.legend()
        self.tx_signal_canvas.draw()
        self.tx_signal_figure.tight_layout(pad=3.0)

    def _plot_constellation(self, received_symbols: np.ndarray, mapping: Dict[Tuple[int, int], complex], num_rx_antennas: int, num_modes: int):
        """Plot the constellation diagram."""
        self.constellation_ax.clear()
        for i in range(num_rx_antennas * num_modes):
            self.constellation_ax.plot(received_symbols[i].real, received_symbols[i].imag, '.', label=f'Primljeni simboli (Prijemnik {i+1})')

        ideal_points = list(mapping.values())
        self.constellation_ax.plot([p.real for p in ideal_points], [p.imag for p in ideal_points], 'r*', markersize=12, markeredgecolor='black', label='Idealni simboli')

        self.constellation_ax.set_xlabel('In-phase komponenta')
        self.constellation_ax.set_ylabel('Quadrature komponenta')
        self.constellation_ax.set_title('Konstelacijski dijagram (QPSK MIMO)')
        self.constellation_ax.grid(True)

        # Create a legend for each receiver
        handles, labels = self.constellation_ax.get_legend_handles_labels()
        for i in range(num_rx_antennas * num_modes):
            handles.append(matplotlib.lines.Line2D([0], [0], linestyle="none", marker='.', markersize=10, markerfacecolor=self.constellation_ax.lines[i].get_color()))
            labels.append(f'Primljeni simboli (Prijemnik {i+1})')
        self.constellation_ax.legend(handles, labels)

        self.constellation_ax.axis('equal')
        self.constellation_canvas.draw()
        self.constellation_figure.tight_layout(pad=3.0)

    def _plot_channel_matrix(self, H: np.ndarray, num_tx_antennas: int, num_rx_antennas: int, num_modes: int):
        """Plot the channel matrix."""
        if not self.channel_matrix_displayed:
            self.channel_ax.clear()
            if num_rx_antennas * num_modes > 0 and num_tx_antennas * num_modes > 0:
                im = self.channel_ax.imshow(np.abs(H), cmap='viridis')
                self.channel_figure.colorbar(im, ax=self.channel_ax, fraction=0.046, pad=0.04, label='Amplituda')
                self.channel_ax.set_xticks(np.arange(num_tx_antennas * num_modes))
                self.channel_ax.set_yticks(np.arange(num_rx_antennas * num_modes))
                self.channel_ax.set_xlabel('Predajni elementi')
                self.channel_ax.set_ylabel('Prijemni elementi')
                self.channel_ax.set_title('Kanalna matrica (H)')
            self.channel_canvas.draw()
            self.channel_matrix_displayed = True

    def _plot_snr_ber(self, snr_db_range: np.ndarray, ber_values: List[float]):
        """Plot SNR vs BER."""
        self.snr_ber_ax.clear()
        self.snr_ber_ax.semilogy(snr_db_range, ber_values, marker='o', linestyle='-')
        self.snr_ber_ax.set_xlabel('SNR (dB)')
        self.snr_ber_ax.set_ylabel('BER')
        self.snr_ber_ax.set_title('Ovisnost BER o SNR')
        self.snr_ber_ax.grid(True)
        self.snr_ber_canvas.draw()

    def _plot_snr_capacity(self, snr_db_range: np.ndarray, capacity_values: List[float]):
        """Plot SNR vs Capacity."""
        self.snr_capacity_ax.clear()
        self.snr_capacity_ax.plot(snr_db_range, capacity_values, marker='o', linestyle='-')
        self.snr_capacity_ax.set_xlabel('SNR (dB)')
        self.snr_capacity_ax.set_ylabel('Kapacitet (bps/Hz)')
        self.snr_capacity_ax.set_title('Ovisnost kapaciteta o SNR')
        self.snr_capacity_ax.grid(True)
        self.snr_capacity_canvas.draw()

    def _plot_noise_impact(self, tx_signals: np.ndarray, noise: np.ndarray, received_symbols: np.ndarray):
        """Plot the impact of noise on the signal."""
        self.noise_impact_ax.clear()
        if tx_signals.size > 0 and noise.size > 0:
            self.noise_impact_ax.plot(np.real(tx_signals[0, :]), label='Odašiljani signal (I komponenta)')
            self.noise_impact_ax.plot(np.imag(tx_signals[0, :]), label='Odašiljani signal (Q komponenta)')
            self.noise_impact_ax.plot(np.real(noise[0, :]), label='Šum (I komponenta)')
            self.noise_impact_ax.plot(np.imag(noise[0, :]), label='Šum (Q komponenta)')
            self.noise_impact_ax.plot(np.real(received_symbols[0, :]), label='Primljeni signal (I komponenta)')
            self.noise_impact_ax.plot(np.imag(received_symbols[0, :]), label='Primljeni signal (Q komponenta)')
            self.noise_impact_ax.set_xlabel('Vrijeme (uzorci)')
            self.noise_impact_ax.set_ylabel('Amplituda')
            self.noise_impact_ax.set_title('Utjecaj šuma na QPSK signal')
            self.noise_impact_ax.legend()
        self.noise_impact_canvas.draw()

    def _plot_detailed_fiber(self, fiber_length: float, attenuation: float):
        """Plot the detailed fiber propagation."""
        self.detailed_fiber_ax.clear()
        distance = np.linspace(0, fiber_length, self.FIBER_LENGTH_POINTS)
        attenuation_linear = 10**(-attenuation / 10)
        signal_power_linear = np.exp(-attenuation_linear * distance) # Simplified model
        signal_power_dbm = 10 * np.log10(signal_power_linear) # Convert to dBm

        # Assume initial noise power is -100 dBm (can be adjusted)
        initial_noise_power_dbm = -100
        noise_power_dbm = initial_noise_power_dbm + 10 * np.log10(1/signal_power_linear) # Noise increases with attenuation

        snr_db = signal_power_dbm - noise_power_dbm

        self.detailed_fiber_ax.plot(distance, signal_power_dbm, label='Snaga signala (dBm)', color='blue')
        self.detailed_fiber_ax.plot(distance, noise_power_dbm, label='Snaga šuma (dBm)', color='red')
        self.detailed_fiber_ax.plot(distance, snr_db, label='SNR (dB)', color='green')
        self.detailed_fiber_ax.set_xlabel('Duljina vlakna (km)')
        self.detailed_fiber_ax.set_ylabel('Snaga (dBm) / SNR (dB)')
        self.detailed_fiber_ax.set_title(f'Prikaz vlakna (Duljina: {fiber_length} km, Atenuacija: {attenuation} dB/km)')
        self.detailed_fiber_ax.grid(True)
        self.detailed_fiber_ax.legend()
        self.detailed_fiber_canvas.draw()

    def _plot_eye_diagram(self, received_symbols: np.ndarray, num_tx_antennas: int, num_modes: int):
        """Plot the eye diagram."""
        self.eye_diagram_ax.clear()
        if received_symbols.size > 0:
            num_symbols_to_plot = min(received_symbols.shape[1], 1000)  # Limit for performance

            # Reshape received symbols to have a single stream for eye diagram
            received_stream = received_symbols.flatten()

            # Calculate the number of traces to plot
            num_traces = len(received_stream) // (2 * num_tx_antennas * num_modes)

            # Create a time axis for the eye diagram
            time_axis = np.arange(0, 2 * num_tx_antennas * num_modes)

            # Plot the eye diagram
            for i in range(num_traces):
                start_index = i * 2 * num_tx_antennas * num_modes
                end_index = start_index + 2 * num_tx_antennas * num_modes
                trace = np.real(received_stream[start_index:end_index])
                self.eye_diagram_ax.plot(time_axis, trace, color='b', alpha=0.5)

            self.eye_diagram_ax.set_xlabel('Vrijeme (simboli)')
            self.eye_diagram_ax.set_ylabel('Amplituda')
            self.eye_diagram_ax.set_title('Eye Dijagram primljenog signala')
            self.eye_diagram_ax.grid(True)
        self.eye_diagram_canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    gui = QPSK_MIMO_GUI(root)
    root.mainloop()
