import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, Label
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from numpy.linalg import svd, det

# Simulation Parameters
        

## @brief Klasa koja implementira tooltip za Tkinter widgete.
class ToolTip:
    """
    @brief Klasa koja implementira tooltip za Tkinter widgete.
    """
    ## @brief Konstruktor za klasu ToolTip.
    ## @param widget Widget na koji se prikači tooltip.
    ## @param text Tekst koji se prikazuje u tooltipu.
    ##
    ## @details Inicijalizira tooltip sa datim widgetom i tekstom, te povezuje događaje prikaza i sakrivanja.
    def __init__(self, widget, text):
        """
        @brief Konstruktor klase ToolTip.
        
        @param widget Tkinter widget na koji se tooltip prikači.
        @param text Tekst koji se prikazuje u tooltipu.
        
        @details Inicijalizira tooltip sa datim widgetom i tekstom, te povezuje događaje miša za prikaz i sakrivanje tooltipa.
        """
        ## @brief Inicijalizira tooltip.
        ## @param widget Widget na koji se prikači tooltip.
        ## @param text Tekst koji se prikazuje u tooltipu.
        ##
        ## @details Inicijalizira tooltip sa datim widgetom i tekstom, te povezuje događaje miša za prikaz i sakrivanje tooltipa.
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)

    ## @brief Prikazuje tooltip prozor.
    ## @param event Događaj koji je pokrenuo prikaz tooltipa (nije obavezno).
    ##
    ## @details Ova metoda izračunava poziciju tooltip prozora i prikazuje ga na ekranu.
    def show(self, event=None):
        """
        @brief Prikazuje tooltip prozor.
        
        @param event Događaj koji je pokrenuo prikaz tooltipa (nije obavezno).
        
        @details Ova metoda izračunava poziciju tooltip prozora i prikazuje ga na ekranu.
        """
        ## @brief Prikazuje tooltip prozor.
        ## @param event Događaj koji je pokrenuo prikaz tooltipa (nije obavezno).
        ##
        ## @details Ova metoda izračunava poziciju tooltip prozora i prikazuje ga na ekranu.
        if self.tooltip_window is None or not self.tooltip_window.winfo_exists():
            x, y, _, _ = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + self.TOOLTIP_OFFSET_X
            y += self.widget.winfo_rooty() + self.TOOLTIP_OFFSET_Y

            self.tooltip_window = Toplevel(self.widget)
            self.tooltip_window.wm_overrideredirect(True)
            self.tooltip_window.wm_geometry(f"+{x}+{y}")

            label = Label(self.tooltip_window, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, padx=1, pady=1)
            label.pack()

    ## @brief Sakriva tooltip prozor.
    ## @param event Događaj koji je pokrenuo sakrivanje tooltipa (nije obavezno).
    ##
    ## @details Ova metoda uništava tooltip prozor ako postoji.
    def hide(self, event=None):
        """
        @brief Sakriva tooltip prozor.
        
        @param event Događaj koji je pokrenuo sakrivanje tooltipa (nije obavezno).
        
        @details Ova metoda uništava tooltip prozor ako postoji.
        """
        ## @brief Sakriva tooltip prozor.
        ## @param event Događaj koji je pokrenuo sakrivanje tooltipa (nije obavezno).
        ##
        ## @details Ova metoda uništava tooltip prozor ako postoji.
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

## @brief Klasa koja implementira GUI za QPSK MIMO simulaciju.
class QPSK_MIMO_GUI:
    ## @brief Konstruktor za klasu QPSK_MIMO_GUI.
    ## @param master Glavni prozor.
    ## @details Inicijalizira glavni prozor i sve GUI elemente za QPSK MIMO simulaciju.
    def __init__(self, master):
        """
        @brief Konstruktor klase QPSK_MIMO_GUI.
        
        @param master Glavni prozor aplikacije.
        
        @details Inicijalizira glavni prozor i sve GUI elemente potrebne za QPSK MIMO simulaciju.
        """
        ## @brief Inicijalizira GUI za QPSK MIMO simulaciju.
        ## @param master Glavni prozor aplikacije.
        ## @details Inicijalizira glavni prozor i sve GUI elemente potrebne za QPSK MIMO simulaciju.
        self.master = master
        master.title("QPSK MIMO Simulacija")
        # master.attributes('-fullscreen', True)  # Start in full-screen

        # Simulation results frame (top right)
        self.results_frame = ttk.LabelFrame(master, text="Rezultati simulacije")
        self.results_frame.pack(padx=10, pady=5, anchor=tk.NE, side=tk.TOP)
        ToolTip(self.results_frame, "Prikazuje rezultate simulacije kao što su BER, SNR i kapacitet.")

        self.ber_label_text = tk.StringVar()
        self.ber_label_text.set("BER: NaN")
        self.ber_label = ttk.Label(self.results_frame, textvariable=self.ber_label_text)
        self.ber_label.pack(padx=5, pady=2)

        self.snr_result_label_text = tk.StringVar()
        self.snr_result_label_text.set("SNR (dB): NaN")
        self.snr_result_label = ttk.Label(self.results_frame, textvariable=self.snr_result_label_text)
        self.snr_result_label.pack(padx=5, pady=2)

        self.capacity_label_text = tk.StringVar()
        self.capacity_label_text.set("Kapacitet (bps/Hz): NaN")
        self.capacity_label = ttk.Label(self.results_frame, textvariable=self.capacity_label_text)
        self.capacity_label.pack(padx=5, pady=2)

        # Input parameters frame
        self.input_frame = ttk.LabelFrame(master, text="Parametri simulacije")
        self.input_frame.pack(padx=10, pady=10, fill=tk.X, anchor=tk.NW, side=tk.TOP)

        # Number of bits
        self.num_bits_label = ttk.Label(self.input_frame, text="Broj bita (100-1000):")
        self.num_bits_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.num_bits_label, "Ukupan broj generisanih bitova za simulaciju.")
        self.num_bits_entry = ttk.Entry(self.input_frame)
        self.num_bits_entry.insert(0, "500")
        self.num_bits_entry.grid(row=0, column=1, padx=5, pady=5)

        # SNR (dB)
        self.snr_label = ttk.Label(self.input_frame, text="SNR (dB) (0-30):")
        self.snr_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.snr_label, "Omjer signala i šuma u decibelima.")
        self.snr_entry = ttk.Entry(self.input_frame)
        self.snr_entry.insert(0, "10")
        self.snr_entry.grid(row=1, column=1, padx=5, pady=5)

        # Broj predajnih antena
        self.num_tx_ant_label = ttk.Label(self.input_frame, text="Broj predajnih antena (1-4):")
        self.num_tx_ant_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.num_tx_ant_label, "Broj predajnih antena u MIMO sistemu.")
        self.num_tx_ant_entry = ttk.Entry(self.input_frame)
        self.num_tx_ant_entry.insert(0, "2")
        self.num_tx_ant_entry.grid(row=2, column=1, padx=5, pady=5)
        self.num_tx_ant_entry.bind("<FocusOut>", self.update_channel_matrix_size)

        # Broj prijemnih antena
        self.num_rx_ant_label = ttk.Label(self.input_frame, text="Broj prijemnih antena (1-4):")
        self.num_rx_ant_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.num_rx_ant_label, "Broj prijemnih antena u MIMO sistemu.")
        self.num_rx_ant_entry = ttk.Entry(self.input_frame)
        self.num_rx_ant_entry.insert(0, "2")
        self.num_rx_ant_entry.grid(row=3, column=1, padx=5, pady=5)
        self.num_rx_ant_entry.bind("<FocusOut>", self.update_channel_matrix_size)

        # Broj modova
        self.num_modes_label = ttk.Label(self.input_frame, text="Broj modova (1-4):")
        self.num_modes_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.num_modes_label, "Broj prostornih modova u vlaknu.")
        self.num_modes_entry = ttk.Entry(self.input_frame)
        self.num_modes_entry.insert(0, "2")
        self.num_modes_entry.grid(row=4, column=1, padx=5, pady=5)
        self.num_modes_entry.bind("<FocusOut>", self.update_channel_matrix_size)

        # Kanalna matrica
        self.channel_label = ttk.Label(self.input_frame, text="Kanalna matrica (H):")
        self.channel_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.channel_label, "Matrica koja opisuje propagaciju signala između predajnih i prijemnih antena/modova.")
        self.channel_matrix_button = ttk.Button(self.input_frame, text="Info", command=self.show_channel_matrix_popup)
        self.channel_matrix_button.grid(row=5, column=1, padx=5, pady=5)
        self.channel_matrix_entries = []
        self.channel_matrix_entry_readonly = True

        # Dužina vlakna (km)
        self.fiber_length_label = ttk.Label(self.input_frame, text="Dužina vlakna (km) (1-1000):")
        self.fiber_length_label.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.fiber_length_label, "Dužina optičkog vlakna u kilometrima.")
        self.fiber_length_entry = ttk.Entry(self.input_frame)
        self.fiber_length_entry.insert(0, "100")
        self.fiber_length_entry.grid(row=6, column=1, padx=5, pady=5)

        # Koeficijent slabljenja (dB/km)
        self.attenuation_label = ttk.Label(self.input_frame, text="Koef. slabljenja (dB/km) (0.1-1):")
        self.attenuation_label.grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.attenuation_label, "Koeficijent slabljenja signala po kilometru vlakna.")
        self.attenuation_entry = ttk.Entry(self.input_frame)
        self.attenuation_entry.insert(0, "0.2")
        self.attenuation_entry.grid(row=7, column=1, padx=5, pady=5)

        # Crosstalk checkbox
        self.crosstalk_var = tk.BooleanVar(value=False)
        self.crosstalk_check = ttk.Checkbutton(self.input_frame, text="Preslušavanje", variable=self.crosstalk_var)
        self.crosstalk_check.grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.crosstalk_check, "Uključuje/isključuje preslušavanje između modova.")

        self.explain_button = ttk.Button(master, text="Objasni koncept", command=self.explain_concept)
        self.explain_button.pack(pady=5, side=tk.RIGHT, padx=10, anchor=tk.NE)

        self.help_button = ttk.Button(master, text="Pomoć", command=self.show_help)
        self.help_button.pack(pady=5, side=tk.RIGHT, padx=10, anchor=tk.NE)

        self.simulate_button = ttk.Button(master, text="Simuliraj", command=self.start_simulation)
        self.simulate_button.pack(pady=5, side=tk.LEFT, padx=10, anchor=tk.NW)

        self.reset_button = ttk.Button(master, text="Resetuj", command=self.reset_simulation)
        self.reset_button.pack(pady=5, side=tk.LEFT, padx=10, anchor=tk.NW)

        # Loading label
        self.loading_label = ttk.Label(master, text="Učitavanje...", font=("Arial", 16))
        self.loading_label.pack(pady=5, side=tk.BOTTOM, anchor=tk.SW, padx=10)
        self.loading_label.pack_forget()  # Initially hide the loading label

        # Notebook for tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10, side=tk.TOP)

        # Tab 1: Transmitted Signal
        self.tx_signal_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tx_signal_tab, text="Odašiljani signal")
        self.tx_signal_figure, (self.tx_signal_ax, self.tx_signal_time_ax) = plt.subplots(1, 2, figsize=(10, 5))
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
        self.channel_figure, (self.channel_ax_mag, self.channel_ax_phase) = plt.subplots(1, 2, figsize=(10, 5))
        self.channel_canvas = FigureCanvasTkAgg(self.channel_figure, master=self.channel_tab)
        self.channel_canvas_widget = self.channel_canvas.get_tk_widget()
        self.channel_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.channel_figure.tight_layout(pad=3.0)

        # Tab 4: Eye Diagram
        self.eye_diagram_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.eye_diagram_tab, text="Eye Dijagram")
        self.eye_diagram_figure, (self.eye_diagram_ax_before, self.eye_diagram_ax_after) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
        self.eye_diagram_canvas = FigureCanvasTkAgg(self.eye_diagram_figure, master=self.eye_diagram_tab)
        self.eye_diagram_canvas_widget = self.eye_diagram_canvas.get_tk_widget()
        self.eye_diagram_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.eye_diagram_figure.tight_layout(pad=3.0)

        # Tab 5: Utjecaj šuma
        self.noise_impact_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.noise_impact_tab, text="Utjecaj šuma na signal")
        self.noise_impact_figure, (self.noise_impact_ax_real, self.noise_impact_ax_imag) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
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

        # Tab 7: Number of Modes vs BER
        self.modes_ber_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.modes_ber_tab, text="Broj modova vs BER")
        self.modes_ber_figure, self.modes_ber_ax = plt.subplots()
        self.modes_ber_canvas = FigureCanvasTkAgg(self.modes_ber_figure, master=self.modes_ber_tab)
        self.modes_ber_canvas_widget = self.modes_ber_canvas.get_tk_widget()
        self.modes_ber_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.modes_ber_figure.tight_layout(pad=3.0)

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
        self.received_symbols = np.array([])
        self.equalized_symbols = np.array([])
        self.update_channel_matrix_size()
        self.channel_matrix_entries = []
        self.channel_matrix_entry_readonly = True
        self.channel_matrix_popup = None
        self.COUPLING_COEFF = 0.01
        self.DMD_COEFF = 0.001
        self.SEED = 42
        self.MIN_NUM_BITS = 100
        self.MAX_NUM_BITS = 1000
        self.MIN_SNR_DB = 0
        self.MAX_SNR_DB = 30
        self.MIN_NUM_ANTENNAS = 1
        self.MAX_NUM_ANTENNAS = 4
        self.MIN_NUM_MODES = 1
        self.MAX_NUM_MODES = 4
        self.MIN_FIBER_LENGTH = 1
        self.MAX_FIBER_LENGTH = 1000
        self.MIN_ATTENUATION = 0.1
        self.MAX_ATTENUATION = 1
        self.EYE_DIAGRAM_SYMBOLS = 2000
        self.CONSTELLATION_MARGIN = 0.2
        self.FIBER_PROPAGATION_POINTS = 100
        self.FIBER_LENGTH_SCALE = 100
        self.CROSSTALK_COEFF = 0.01
        self.TOOLTIP_OFFSET_X = 25
        self.TOOLTIP_OFFSET_Y = 25

        # Initial draw of plots
        self.show_all_plots()

    ## @brief Sakriva sve grafove.
    ## @details Ova metoda sakriva sve grafove u GUI.
    def hide_all_plots(self):
        """
        @brief Sakriva sve grafove.
        
        @details Ova metoda sakriva sve grafove u GUI.
        """

    ## @brief Prikazuje sve grafove.
    ## @details Ova metoda prikazuje sve grafove u GUI.
    def show_all_plots(self):
        """
        @brief Prikazuje sve grafove.
        
        @details Ova metoda prikazuje sve grafove u GUI.
        """
        self.tx_signal_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.constellation_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.channel_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.eye_diagram_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.noise_impact_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.snr_ber_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.modes_ber_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.detailed_fiber_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    ## @brief Pokreće proces simulacije.
    ## @details Ova metoda inicira simulaciju postavljanjem unosa matrice kanala na uređivanje i pozivanjem metode simulacije.
    def start_simulation(self):
        """
        @brief Pokreće proces simulacije.
        
        @details Ova metoda inicira simulaciju postavljanjem unosa matrice kanala na uređivanje i pozivanjem metode simulacije.
        """
        
        self.loading_label.pack(side=tk.BOTTOM, anchor=tk.SW, padx=10, pady=5)
        self.master.update()
        self.simulate_button.config(state=tk.DISABLED)
        try:
            self.simulate()
        except Exception as e:
            messagebox.showerror("Greška", f"Došlo je do greške tijekom simulacije: {e}")
        finally:
            self.simulate_button.config(state=tk.NORMAL)

    ## @brief Generiše matricu kanala.
    ## @param num_tx_modes Broj predajnih modova.
    ## @param num_rx_modes Broj prijemnih modova.
    ## @param fiber_length Dužina vlakna u km.
    ## @param seed Slučajni seed za ponovljivost.
    ## @param coupling_coeff Koeficijent sprezanja.
    ## @param dmd_coeff Koeficijent diferencijalnog kašnjenja moda.
    ## @return Generisana matrica kanala.
    ## @details Ova metoda generiše matricu kanala na osnovu broja predajnih i prijemnih modova, dužine vlakna i ostalih parametara.
    def generate_channel_matrix(self, num_tx_modes, num_rx_modes, fiber_length, seed=None, coupling_coeff=0.01, dmd_coeff=0.001):
        """
        @brief Generiše matricu kanala.
        
        @param num_tx_modes Broj predajnih modova.
        @param num_rx_modes Broj prijemnih modova.
        @param fiber_length Dužina vlakna u km.
        @param seed Slučajni seed za ponovljivost.
        @param coupling_coeff Koeficijent sprezanja.
        @param dmd_coeff Koeficijent diferencijalnog kašnjenja moda.
        @return Generisana matrica kanala.
        @details Ova metoda generiše matricu kanala na osnovu broja predajnih i prijemnih modova, dužine vlakna i ostalih parametara.
        """
        if seed is not None:
            np.random.seed(seed)

        # More realistic mode coupling matrix
        coupling_matrix = coupling_coeff * np.exp(-((np.arange(num_rx_modes)[:, None] - np.arange(num_tx_modes)) / 2)**2) * (np.random.randn(num_rx_modes, num_tx_modes) + 1j * np.random.randn(num_rx_modes, num_tx_modes))
        
        # Ensure the diagonal is 1
        np.fill_diagonal(coupling_matrix, 1)
        
        # Differential mode delay (DMD) as a phase shift
        dmd_matrix = np.exp(-1j * dmd_coeff * (np.arange(num_rx_modes)[:, None] - np.arange(num_tx_modes))**2)

        H = np.dot(dmd_matrix, coupling_matrix)
        return H

    ## @brief Izračunava Bit Error Rate (BER).
    ## @param tx_bits Predajni bitovi.
    ## @param demodulated_bits Demodulirani bitovi.
    ## @return Izračunata BER vrijednost.
    ## @details Ova metoda izračunava BER uspoređujući predajne i demodulirane bitove.
    def calculate_ber(self, tx_bits, demodulated_bits):
        """
        @brief Calculates the Bit Error Rate (BER).
        
        @param tx_bits The transmitted bits.
        @param demodulated_bits The demodulated bits.
        @return The calculated BER value.
        @details This method calculates the BER by comparing the transmitted and demodulated bits.
        """
        return self._calculate_ber(tx_bits, demodulated_bits)

    ## @brief Izračunava Bit Error Rate (BER).
    ## @param tx_bits Predajni bitovi.
    ## @param demodulated_bits Demodulirani bitovi.
    ## @return Izračunata BER vrijednost.
    ## @details Ova metoda izračunava BER uspoređujući predajne i demodulirane bitove.
    def _calculate_ber(self, tx_bits, demodulated_bits):
        """
        @brief Izračunava Bit Error Rate (BER).
        
        @param tx_bits Predajni bitovi.
        @param demodulated_bits Demodulirani bitovi.
        @return Izračunata BER vrijednost.
        @details Ova metoda izračunava BER uspoređujući predajne i demodulirane bitove.
        """
        if len(tx_bits) == 0:
            return np.nan
        if len(demodulated_bits) == 0:
            return 0
        min_len = min(len(tx_bits), len(demodulated_bits))
        return np.sum(np.array(tx_bits[:min_len]) != np.array(demodulated_bits[:min_len])) / len(tx_bits)

    ## @brief Demodulira primljene simbole u bitove.
    ## @param received_symbols Primljeni simboli.
    ## @param mapping Rječnik QPSK mapiranja.
    ## @param inverse_mapping Inverzni rječnik QPSK mapiranja.
    ## @return Demodulirani bitovi.
    ## @details Ova metoda demodulira primljene simbole u bitove koristeći pristup najbližeg susjeda.
    def _demodulate_symbols(self, received_symbols, mapping, inverse_mapping):
        """
        @brief Demodulira primljene simbole u bitove.
        
        @param received_symbols Primljeni simboli.
        @param mapping Rječnik QPSK mapiranja.
        @param inverse_mapping Inverzni rječnik QPSK mapiranja.
        @return Demodulirani bitovi.
        @details Ova metoda demodulira primljene simbole u bitove koristeći pristup najbližeg susjeda.
        """
        ## @brief Demodulira primljene simbole u bitove.
        ## @param received_symbols Primljeni simboli.
        ## @param mapping Rječnik QPSK mapiranja.
        ## @param inverse_mapping Inverzni rječnik QPSK mapiranja.
        ## @return Demodulirani bitovi.
        ## @details Ova metoda demodulira primljene simbole u bitove koristeći pristup najbližeg susjeda.
        ref_symbols = np.array(list(mapping.values()))
        
        # Reshape received symbols to have the same number of rows as ref_symbols
        received_symbols_reshaped = received_symbols.reshape(received_symbols.shape[0], -1)
        
        # Calculate distances between each received symbol and all reference symbols
        distances = np.abs(received_symbols_reshaped[:, :, np.newaxis] - ref_symbols)**2
        
        # Find the index of the closest reference symbol for each received symbol
        closest_indices = np.argmin(distances, axis=2)
        
        # Map the closest indices to the corresponding bits
        demodulated_bits = []
        for i in range(closest_indices.shape[0]):
            for index in closest_indices[i]:
                closest_symbol = ref_symbols[index]
                demodulated_bits.extend(inverse_mapping[closest_symbol])
        return demodulated_bits

    ## @brief Izračunava Bit Error Rate (BER) za dati SNR.
    ## @param tx_bits Predajni bitovi.
    ## @param received_symbols Primljeni simboli.
    ## @param mapping Rječnik QPSK mapiranja.
    ## @param inverse_mapping Inverzni rječnik QPSK mapiranja.
    ## @return Izračunata BER vrijednost.
    ## @details Ova metoda izračunava BER uspoređujući predajne i demodulirane bitove.
    def calculate_ber_for_snr(self, tx_bits, received_symbols, mapping, inverse_mapping):
        """
        @brief Izračunava Bit Error Rate (BER) za dati SNR.
        
        @param tx_bits Predajni bitovi.
        @param received_symbols Primljeni simboli.
        @param mapping Rječnik QPSK mapiranja.
        @param inverse_mapping Inverzni rječnik QPSK mapiranja.
        @return Izračunata BER vrijednost.
        @details Ova metoda izračunava BER uspoređujući predajne i demodulirane bitove.
        """
        ## @brief Izračunava Bit Error Rate (BER) za dati SNR.
        ## @param tx_bits Predajni bitovi.
        ## @param received_symbols Primljeni simboli.
        ## @param mapping Rječnik QPSK mapiranja.
        ## @param inverse_mapping Inverzni rječnik QPSK mapiranja.
        ## @return Izračunata BER vrijednost.
        ## @details Ova metoda izračunava BER uspoređujući predajne i demodulirane bitove.
        demodulated_bits_ber = self._demodulate_symbols(received_symbols, mapping, inverse_mapping)
        return self._calculate_ber(tx_bits, demodulated_bits_ber)

    ## @brief Prikazuje prozor za pomoć sa uputama o korištenju simulacije.
    ## @details Ova metoda kreira i prikazuje prozor za pomoć sa uputama o korištenju simulacije.
    def show_help(self):
        """
        @brief Prikazuje prozor za pomoć sa uputama o korištenju simulacije.
        
        @details Ova metoda kreira i prikazuje prozor za pomoć sa uputama o korištenju simulacije.
        """
        ## @brief Prikazuje prozor za pomoć sa uputama o korištenju simulacije.
        ## @details Ova metoda kreira i prikazuje prozor za pomoć sa uputama o korištenju simulacije.
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
        - **Matrica kanala:** Prikazuje matricu kanala H.
        - **Eye Dijagram:** Prikazuje eye dijagram primljenog signala.
        - **Utjecaj šuma na signal:** Prikazuje utjecaj šuma na odašiljani i primljeni signal.
        - **SNR vs BER:** Prikazuje ovisnost BER o SNR.
        - **Broj modova vs BER:** Prikazuje ovisnost BER o broju modova.
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

    ## @brief Prikazuje prozor koji objašnjava koncept QPSK MIMO modulacije u višemodnom vlaknu.
    ## @details Ova metoda kreira i prikazuje prozor sa detaljnim objašnjenjem QPSK MIMO koncepta u višemodnom vlaknu.
    def explain_concept(self):
        """
        @brief Prikazuje prozor koji objašnjava koncept QPSK MIMO modulacije u višemodnom vlaknu.
        
        @details Ova metoda kreira i prikazuje prozor sa detaljnim objašnjenjem QPSK MIMO koncepta u višemodnom vlaknu.
        """
        ## @brief Prikazuje prozor koji objašnjava koncept QPSK MIMO modulacije u višemodnom vlaknu.
        ## @details Ova metoda kreira i prikazuje prozor sa detaljnim objašnjenjem QPSK MIMO koncepta u višemodnom vlaknu.
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
        self.channel_ax_mag.clear()
        self.channel_ax_phase.clear()
        self.channel_canvas.draw()
        self.snr_ber_ax.clear()
        self.snr_ber_canvas.draw()
        self.modes_ber_ax.clear()
        self.modes_ber_canvas.draw()
        self.noise_impact_ax.clear()
        self.noise_impact_canvas.draw()
        self.detailed_fiber_ax.clear()
        self.detailed_fiber_canvas.draw()
        # Close all figures
        if hasattr(self, 'tx_signal_figure') and self.tx_signal_figure:
            plt.close(self.tx_signal_figure)
        if hasattr(self, 'constellation_figure') and self.constellation_figure:
            plt.close(self.constellation_figure)
        if hasattr(self, 'channel_figure') and self.channel_figure:
            plt.close(self.channel_figure)
        if hasattr(self, 'eye_diagram_figure') and self.eye_diagram_figure:
            plt.close(self.eye_diagram_figure)
        if hasattr(self, 'noise_impact_figure') and self.noise_impact_figure:
            plt.close(self.noise_impact_figure)
        if hasattr(self, 'snr_ber_figure') and self.snr_ber_figure:
            plt.close(self.snr_ber_figure)
        if hasattr(self, 'modes_ber_figure') and self.modes_ber_figure:
            plt.close(self.modes_ber_figure)
        if hasattr(self, 'detailed_fiber_figure') and self.detailed_fiber_figure:
            plt.close(self.detailed_fiber_figure)

        # Clear all plots
        self.tx_signal_ax.clear()
        self.tx_signal_canvas.draw()
        self.constellation_ax.clear()
        self.constellation_canvas.draw()
        self.channel_ax_mag.clear()
        self.channel_ax_phase.clear()
        self.channel_canvas.draw()
        self.snr_ber_ax.clear()
        self.snr_ber_canvas.draw()
        self.modes_ber_ax.clear()
        self.modes_ber_canvas.draw()
        self.noise_impact_ax_real.clear()
        self.noise_impact_ax_imag.clear()
        self.noise_impact_canvas.draw()
        self.detailed_fiber_ax.clear()
        self.detailed_fiber_canvas.draw()
        self.eye_diagram_ax_before.clear()
        self.eye_diagram_ax_after.clear()
        self.eye_diagram_canvas.draw()
        if self.fiber_propagation_ax:
            self.fiber_propagation_ax.clear()
            self.fiber_propagation_canvas.draw()
        self.channel_matrix_entry_readonly = True
        self.update_channel_matrix_entry_state()


        # Reset result labels
        self.ber_label_text.set("BER: NaN")
        self.snr_result_label_text.set("SNR (dB): NaN")
        self.capacity_label_text.set("Kapacitet (bps/Hz): NaN")
        self.channel_matrix_displayed = False

    ## @brief Resetuje sve grafove i rezultate simulacije.
    ## @details Ova metoda briše sve grafove i resetuje rezultate simulacije na početne vrijednosti.
    def reset_simulation(self):
        """
        @brief Resetuje sve grafove i rezultate simulacije.
        
        @details Ova metoda briše sve grafove i resetuje rezultate simulacije na početne vrijednosti.
        """
        ## @brief Resetuje sve grafove i rezultate simulacije.
        ## @details Ova metoda briše sve grafove i resetuje rezultate simulacije na početne vrijednosti.
        # Clear all plots
        self.tx_signal_ax.clear()
        self.tx_signal_canvas.draw()
        self.constellation_ax.clear()
        self.constellation_canvas.draw()
        self.channel_ax_mag.clear()
        self.channel_ax_phase.clear()
        self.channel_canvas.draw()
        self.snr_ber_ax.clear()
        self.snr_ber_canvas.draw()
        self.modes_ber_ax.clear()
        self.modes_ber_canvas.draw()
        self.noise_impact_ax_real.clear()
        self.noise_impact_ax_imag.clear()
        self.noise_impact_canvas.draw()
        self.detailed_fiber_ax.clear()
        self.detailed_fiber_canvas.draw()
        self.eye_diagram_ax_before.clear()
        self.eye_diagram_ax_after.clear()
        self.eye_diagram_canvas.draw()
        self.tx_signal_time_ax.clear()
        self.tx_signal_canvas.draw()
        if self.fiber_propagation_ax:
            self.fiber_propagation_ax.clear()
            self.fiber_propagation_canvas.draw()
        self.channel_matrix_entry_readonly = True
        self.update_channel_matrix_entry_state()


        # Reset result labels
        self.ber_label_text.set("BER: NaN")
        self.snr_result_label_text.set("SNR (dB): NaN")
        self.capacity_label_text.set("Kapacitet (bps/Hz): NaN")
        self.channel_matrix_displayed = False

    ## @brief Ažurira veličinu matrice kanala na osnovu broja modova, predajnih i prijemnih antena.
    ## @param event Događaj koji je pokrenuo ažuriranje.
    ## @details Ova metoda ažurira veličinu matrice kanala na osnovu broja modova, predajnih i prijemnih antena unesenih u GUI.
    def update_channel_matrix_size(self, event=None):
        """
        @brief Ažurira veličinu matrice kanala na osnovu broja modova, predajnih i prijemnih antena.
        
        @param event Događaj koji je pokrenuo ažuriranje.
        @details Ova metoda ažurira veličinu matrice kanala na osnovu broja modova, predajnih i prijemnih antena unesenih u GUI.
        """
        ## @brief Ažurira veličinu matrice kanala na osnovu broja modova, predajnih i prijemnih antena.
        ## @param event Događaj koji je pokrenuo ažuriranje.
        ## @details Ova metoda ažurira veličinu matrice kanala na osnovu broja modova, predajnih i prijemnih antena unesenih u GUI.
        try:
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
            
            # Clear the channel matrix plot before updating the size
            self.channel_ax_mag.clear()
            self.channel_ax_phase.clear()
            
            new_matrix_size = (num_rx_antennas * num_modes, num_tx_antennas * num_modes)
            
            # Adjust figure size to fit the new matrix
            self.channel_figure.set_size_inches(max(new_matrix_size[1] * 1.2, 6), max(new_matrix_size[0] * 0.8, 4))
            self.channel_figure.tight_layout()
            default_matrix = np.zeros(new_matrix_size, dtype=complex)
            np.fill_diagonal(default_matrix, 1)
            self.create_channel_matrix_entries(default_matrix.tolist(), None, None)
        except ValueError:
            messagebox.showerror("Greška", "Neispravan unos za broj modova.")
            self.num_modes_entry.delete(0, tk.END)
            self.num_modes_entry.insert(0, "1")
            self.update_channel_matrix_entry_state()

    ## @brief Izvršava QPSK MIMO simulaciju.
    ## @details Ova metoda preuzima parametre simulacije iz GUI, izvršava simulaciju i ažurira GUI sa rezultatima.
    def simulate(self):
        """
        @brief Izvršava QPSK MIMO simulaciju.
        
        @details Ova metoda preuzima parametre simulacije iz GUI, izvršava simulaciju i ažurira GUI sa rezultatima.
        """
        ## @brief Izvršava QPSK MIMO simulaciju.
        ## @details Ova metoda preuzima parametre simulacije iz GUI, izvršava simulaciju i ažurira GUI sa rezultatima.
        # Dohvati parametre simulacije iz GUI
        try:
            num_bits = int(self.num_bits_entry.get())
        except ValueError:
            messagebox.showerror("Greška", "Broj bita mora biti cijeli broj.")
            return
        if not self.MIN_NUM_BITS <= num_bits <= self.MAX_NUM_BITS:
            messagebox.showerror("Greška", f"Broj bita mora biti između {self.MIN_NUM_BITS} i {self.MAX_NUM_BITS}.")
            return

        try:
            snr_db = float(self.snr_entry.get())
        except ValueError:
             messagebox.showerror("Greška", "SNR mora biti broj.")
             return
        if not self.MIN_SNR_DB <= snr_db <= self.MAX_SNR_DB:
            messagebox.showerror("Greška", f"SNR mora biti između {self.MIN_SNR_DB} i {self.MAX_SNR_DB} dB.")
            return

        try:
            fiber_length = float(self.fiber_length_entry.get())
        except ValueError:
            messagebox.showerror("Greška", "Dužina vlakna mora biti broj.")
            return
        if not self.MIN_FIBER_LENGTH <= fiber_length <= self.MAX_FIBER_LENGTH:
            messagebox.showerror("Greška", f"Dužina vlakna mora biti između {self.MIN_FIBER_LENGTH} i {self.MAX_FIBER_LENGTH} km.")
            return

        try:
            attenuation = float(self.attenuation_entry.get())
        except ValueError:
            messagebox.showerror("Greška", "Koeficijent slabljenja mora biti broj.")
            return
        if not self.MIN_ATTENUATION <= attenuation <= self.MAX_ATTENUATION:
            messagebox.showerror("Greška", f"Koeficijent slabljenja mora biti između {self.MIN_ATTENUATION} i {self.MAX_ATTENUATION} dB/km.")
            return

        num_modes = int(self.num_modes_entry.get())
        if not self.MIN_NUM_MODES <= num_modes <= self.MAX_NUM_MODES:
            messagebox.showerror("Greška", "Broj modova mora biti između 1 i 4.")
            return

        num_tx_antennas = int(self.num_tx_ant_entry.get())
        if not self.MIN_NUM_ANTENNAS <= num_tx_antennas <= self.MAX_NUM_ANTENNAS:
            messagebox.showerror("Greška", "Broj predajnih antena mora biti između 1 i 4.")
            return
        num_rx_antennas = int(self.num_rx_ant_entry.get())
        if not self.MIN_NUM_ANTENNAS <= num_rx_antennas <= self.MAX_NUM_ANTENNAS:
            messagebox.showerror("Greška", "Broj prijemnih antena mora biti između 1 i 4.")
            return
        
        self.channel_matrix_entry_readonly = True
        self.update_channel_matrix_entry_state()

        num_tx_modes = num_tx_antennas * num_modes
        num_rx_modes = num_rx_antennas * num_modes
        coupling_coeff = self.COUPLING_COEFF * np.sqrt(fiber_length / self.FIBER_LENGTH_SCALE)  # Non-linear scaling
        dmd_coeff = self.DMD_COEFF * np.sqrt(fiber_length / self.FIBER_LENGTH_SCALE) # Non-linear scaling
        
        H = self.get_channel_matrix_from_entries()
        if H.size == 0:
            H = self.generate_channel_matrix(num_tx_modes, num_rx_modes, fiber_length, seed=self.SEED, coupling_coeff=coupling_coeff, dmd_coeff=dmd_coeff)

        bits = np.random.randint(0, 2, num_bits)

        # QPSK Modulacija
        mapping = {
            (0, 0): complex(1/np.sqrt(2), 1/np.sqrt(2)),  # 00 -> 1+j
            (0, 1): complex(-1/np.sqrt(2), 1/np.sqrt(2)), # 01 -> -1+j
            (1, 0): complex(1/np.sqrt(2), -1/np.sqrt(2)), # 10 -> 1-j
            (1, 1): complex(-1/np.sqrt(2), -1/np.sqrt(2))  # 11 -> -1-j
        }
        inverse_mapping = {v: k for k, v in mapping.items()}
        
        # Pre-calculate QPSK symbols
        qpsk_symbols = np.array(list(mapping.values()))
        
        # Generate transmit bits
        tx_bits = bits[:(len(bits) // 2) * 2]
        
        # Map bits to QPSK symbols
        qpsk_symbol_indices = np.array(tx_bits).reshape(-1, 2)
        qpsk_symbols_mapped = np.array([mapping[tuple(index)] for index in qpsk_symbol_indices])

        # MIMO dio
        mode_bits = np.random.randint(0, 2, (num_tx_modes, len(tx_bits)))
        mode_symbols = np.array([mapping[tuple(bits)] for bits in mode_bits.reshape(num_tx_modes, -1, 2).transpose(0, 2, 1).reshape(-1, 2)]).reshape(num_tx_modes, -1)
        tx_signals = mode_symbols

        # Apply channel matrix
        rx_signals = np.dot(H, tx_signals)

        # Add crosstalk
        if self.crosstalk_var.get():
            # Create a matrix of random complex numbers for crosstalk
            crosstalk_matrix = self.CROSSTALK_COEFF * (np.random.randn(num_rx_modes, num_rx_modes) + 1j * np.random.randn(num_rx_modes, num_rx_modes))
            
            # Set diagonal elements to 0 to avoid self-crosstalk
            np.fill_diagonal(crosstalk_matrix, 0)
            
            # Apply crosstalk using matrix multiplication
            rx_signals += np.dot(crosstalk_matrix, rx_signals)

        # Calculate signal power before adding noise
        signal_power = np.mean(np.abs(rx_signals)**2)

        # Dodavanje šuma (AWGN)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(*rx_signals.shape) + 1j * np.random.randn(*rx_signals.shape))
        self.received_symbols = rx_signals + noise

        # MMSE Equalizer
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Calculate the MMSE equalizer matrix using pseudo-inverse
        try:
            mmse_matrix = np.dot(np.conjugate(H).T, np.linalg.pinv(np.dot(H, np.conjugate(H).T) + (noise_power/signal_power) * np.eye(H.shape[0])))
            self.equalized_symbols = np.dot(mmse_matrix, self.received_symbols)
        except:
            self.equalized_symbols = self.received_symbols

        # Demodulacija (nearest neighbor)
        demodulated_bits = self._demodulate_symbols(self.equalized_symbols, mapping, inverse_mapping)

        # Proračun BER
        if len(tx_bits) > 0:
            ber_point = np.mean(np.array(tx_bits) != np.array(demodulated_bits[:len(tx_bits)]))
            self.ber_label_text.set(f"BER: {ber_point:.4f}")
        else:
            self.ber_label_text.set("BER: NaN")

        self.snr_result_label_text.set(f"SNR (dB): {snr_db:.2f}")

        # Calculate MSE
        mse = np.mean(np.abs(self.equalized_symbols - tx_signals)**2)

        # Calculate EVM
        evm = np.sqrt(np.mean(np.abs(self.equalized_symbols - tx_signals)**2) / np.mean(np.abs(tx_signals)**2))
        self.capacity_label_text.set(f"Kapacitet (bps/Hz): NaN")

        # Prikaz odasiljanog signala (prva antena)
        self.tx_signal_ax.clear()
        self.tx_signal_ax.plot(qpsk_symbols.real, qpsk_symbols.imag, 'o', label='Odašiljani simboli')
        self.tx_signal_ax.set_xlabel('In-phase')
        self.tx_signal_ax.set_ylabel('Quadrature')
        self.tx_signal_ax.set_title('Odašiljani signal (Konstelacija)')
        
        # Set plot limits
        self.tx_signal_ax.set_xlim(-2, 2)
        self.tx_signal_ax.set_ylim(-0.75, 0.75)
        
        self.tx_signal_ax.grid(True)
        self.tx_signal_ax.legend()

        # Time domain plot
        self.tx_signal_time_ax.clear()
        time = np.arange(len(qpsk_symbols))
        self.tx_signal_time_ax.plot(time, np.real(qpsk_symbols), label='I komponenta')
        self.tx_signal_time_ax.plot(time, np.imag(qpsk_symbols), label='Q komponenta')
        self.tx_signal_time_ax.set_xlabel('Vrijeme (simboli)')
        self.tx_signal_time_ax.set_ylabel('Amplituda')
        self.tx_signal_time_ax.set_title('Odašiljani signal (Vrijeme)')
        self.tx_signal_time_ax.grid(True)
        self.tx_signal_time_ax.legend()

        self.tx_signal_canvas.draw()
        self.tx_signal_figure.tight_layout(pad=3.0)

        # Prikaz konstelacije
        self.constellation_ax.clear()
        
        ideal_points = list(mapping.values())
        self.constellation_ax.plot([p.real for p in ideal_points], [p.imag for p in ideal_points], 'r*', markersize=10, label='Idealni simboli')
        
        if self.received_symbols.size > 0:
            num_rx_modes = int(self.num_rx_ant_entry.get()) * int(self.num_modes_entry.get())
            for i in range(num_rx_modes):
                self.constellation_ax.plot(self.received_symbols[i].real, self.received_symbols[i].imag, '.', label=f'Primljeni simboli (Prijemnik {i+1})')

            # Calculate plot limits with a margin
            all_real = np.concatenate([self.received_symbols[i].real for i in range(num_rx_modes)])
            all_imag = np.concatenate([self.received_symbols[i].imag for i in range(num_rx_modes)])
            
            margin_x = self.CONSTELLATION_MARGIN * (max(all_real) - min(all_real))
            margin_y = self.CONSTELLATION_MARGIN * (max(all_imag) - min(all_imag))
            
            self.constellation_ax.set_xlim(min(all_real) - margin_x, max(all_real) + margin_x)
            self.constellation_ax.set_ylim(min(all_imag) - margin_y, max(all_imag) + margin_y)
        else:
            self.constellation_ax.set_xlim(-2, 2)
            self.constellation_ax.set_ylim(-2, 2)
        
        self.constellation_ax.set_xlabel('In-phase')
        self.constellation_ax.set_ylabel('Quadrature')
        self.constellation_ax.set_title('Konstelacijski dijagram (QPSK MIMO)')
        self.constellation_ax.grid(True)
        self.constellation_ax.legend()
        self.constellation_canvas.draw()
        self.constellation_figure.tight_layout(pad=3.0)

        # Prikaz matrice kanala
        if num_rx_antennas * num_modes > 0 and num_tx_antennas * num_modes > 0:
            # Magnitude plot
            im_mag = self.channel_ax_mag.imshow(np.abs(H), cmap='viridis')
            self.channel_figure.colorbar(im_mag, ax=self.channel_ax_mag, fraction=0.046, pad=0.04, label='Amplituda')
            self.channel_ax_mag.set_xticks(np.arange(num_tx_antennas * num_modes))
            self.channel_ax_mag.set_yticks(np.arange(num_rx_antennas * num_modes))
            self.channel_ax_mag.set_xlabel('Predajni elementi')
            self.channel_ax_mag.set_ylabel('Prijemni elementi')
            self.channel_ax_mag.set_title('Magnituda kanalne matrice (H)')
            self.channel_ax_mag.set_xlim(0, num_tx_antennas * num_modes - 1)
            self.channel_ax_mag.set_ylim(num_rx_antennas * num_modes - 1, 0)

            # Phase plot
            im_phase = self.channel_ax_phase.imshow(np.angle(H), cmap='twilight')
            self.channel_figure.colorbar(im_phase, ax=self.channel_ax_phase, fraction=0.046, pad=0.04, label='Faza (rad)')
            self.channel_ax_phase.set_xticks(np.arange(num_tx_antennas * num_modes))
            self.channel_ax_phase.set_yticks(np.arange(num_rx_antennas * num_modes))
            self.channel_ax_phase.set_xlabel('Predajni elementi')
            self.channel_ax_phase.set_ylabel('Prijemni elementi')
            self.channel_ax_phase.set_title('Faza kanalne matrice (H)')
            self.channel_ax_phase.set_xlim(0, num_tx_antennas * num_modes - 1)
            self.channel_ax_phase.set_ylim(num_rx_antennas * num_modes - 1, 0)
        self.channel_canvas.draw()
        
        # SNR vs BER plot
        snr_db_range = np.linspace(0, 20, 10)
        num_noise_realizations = 10  # Number of noise realizations for averaging
        ber_values = []
        for current_snr_db in snr_db_range:
            snr_linear = 10**(current_snr_db / 10)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power / 2)
            
            # Generate noise for all realizations at once
            noise = noise_std * (np.random.randn(num_noise_realizations, *rx_signals.shape) + 1j * np.random.randn(num_noise_realizations, *rx_signals.shape))
            
            received_symbols_ber = rx_signals + noise
            
            # Calculate BER for all noise realizations and average
            ber_sum = np.mean([self.calculate_ber_for_snr(tx_bits, received_symbols_ber[i], mapping, inverse_mapping) for i in range(num_noise_realizations)])
            ber_values.append(ber_sum)

        self.snr_ber_ax.clear()
        self.snr_ber_ax.semilogy(snr_db_range, ber_values, marker='o', linestyle='-')
        self.snr_ber_ax.set_xlabel('SNR (dB)')
        self.snr_ber_ax.set_ylabel('BER')
        self.snr_ber_ax.set_title('Ovisnost BER o SNR')
        self.snr_ber_ax.grid(True)
        self.snr_ber_canvas.draw()

        # Number of Modes vs BER plot
        num_modes_values = [1, 2, 3, 4]  # Different numbers of modes
        fiber_lengths = [100, 500, 1000]  # Fixed fiber lengths in km
        
        self.modes_ber_ax.clear()
        for fiber_length in fiber_lengths:
            bits = np.random.randint(0, 2, int(self.num_bits_entry.get()))
            ber_values = []
            for num_modes in num_modes_values:
                num_tx_modes = int(self.num_tx_ant_entry.get()) * num_modes
                num_rx_modes = int(self.num_rx_ant_entry.get()) * num_modes
                coupling_coeff = self.COUPLING_COEFF * np.sqrt(fiber_length / self.FIBER_LENGTH_SCALE)  # Non-linear scaling
                dmd_coeff = self.DMD_COEFF * np.sqrt(fiber_length / self.FIBER_LENGTH_SCALE) # Non-linear scaling
                
                
                mapping = {
                    (0, 0): complex(1/np.sqrt(2), 1/np.sqrt(2)),  # 00 -> 1+j
                    (0, 1): complex(-1/np.sqrt(2), 1/np.sqrt(2)), # 01 -> -1+j
                    (1, 0): complex(1/np.sqrt(2), -1/np.sqrt(2)), # 10 -> 1-j
                    (1, 1): complex(-1/np.sqrt(2), -1/np.sqrt(2))  # 11 -> -1-j
                }
                inverse_mapping = {v: k for k, v in mapping.items()}
                
                qpsk_symbols = []
                tx_bits = []
                for i in range(0, len(bits), 2):
                    if i + 1 < len(bits):
                        tx_bits.extend([bits[i], bits[i+1]])
                        symbol = mapping[(bits[i], bits[i+1])]
                        qpsk_symbols.append(symbol)

                qpsk_symbols = np.array(qpsk_symbols)
                
                tx_signals = np.zeros((num_tx_modes, len(qpsk_symbols)), dtype=complex)
                for mode_idx in range(num_tx_modes):
                    mode_bits = np.random.randint(0, 2, len(tx_bits))
                    mode_qpsk_symbols = []
                    for i in range(0, len(mode_bits), 2):
                        if i + 1 < len(mode_bits):
                            symbol = mapping[(mode_bits[i], mode_bits[i+1])]
                            mode_qpsk_symbols.append(symbol)
                    tx_signals[mode_idx, :] = np.array(mode_qpsk_symbols)
                
                # Generate channel matrix with correct dimensions
                H = self.generate_channel_matrix(num_tx_modes, num_rx_modes, fiber_length, seed=42, coupling_coeff=coupling_coeff, dmd_coeff=dmd_coeff)
                
                rx_signals = np.dot(H, tx_signals)
                signal_power = np.mean(np.abs(rx_signals)**2)
                snr_db = float(self.snr_entry.get())
                snr_linear = 10**(snr_db / 10)
                noise_power = signal_power / snr_linear
                noise_std = np.sqrt(noise_power / 2)
                noise = noise_std * (np.random.randn(*rx_signals.shape) + 1j * np.random.randn(*rx_signals.shape))
                received_symbols_ber = rx_signals + noise
                ber = self.calculate_ber_for_snr(tx_bits, received_symbols_ber, mapping, inverse_mapping)
                ber_values.append(ber)
            self.modes_ber_ax.semilogy(num_modes_values, ber_values, marker='o', linestyle='-', label=f'{fiber_length} km')
        
        self.modes_ber_ax.set_xlabel('Broj modova')
        self.modes_ber_ax.set_ylabel('BER')
        self.modes_ber_ax.set_title('Ovisnost BER o broju modova')
        self.modes_ber_ax.grid(True)
        self.modes_ber_ax.legend()
        self.modes_ber_canvas.draw()

        # Utjecaj šuma na signal
        self.noise_impact_ax_real.clear()
        self.noise_impact_ax_imag.clear()
        if qpsk_symbols.size > 0 and noise.size > 0:
            time = np.arange(len(tx_signals[0, :]))
            
            # Plot real components
            self.noise_impact_ax_real.plot(time, np.real(tx_signals[0, :]), label='Odašiljani signal (I)')
            self.noise_impact_ax_real.plot(time, np.real(noise[0, :]), label='Šum (I)')
            self.noise_impact_ax_real.plot(time, np.real(self.received_symbols[0, :]), label='Primljeni signal (I)')
            self.noise_impact_ax_real.set_ylabel('Amplituda')
            self.noise_impact_ax_real.set_title('Realne komponente')
            self.noise_impact_ax_real.grid(True)
            
            # Plot imaginary components
            self.noise_impact_ax_imag.plot(time, np.imag(tx_signals[0, :]), label='Odašiljani signal (Q)')
            self.noise_impact_ax_imag.plot(time, np.imag(noise[0, :]), label='Šum (Q)')
            self.noise_impact_ax_imag.plot(time, np.imag(self.received_symbols[0, :]), label='Primljeni signal (Q)')
            self.noise_impact_ax_imag.set_xlabel('Vrijeme (uzorci)')
            self.noise_impact_ax_imag.set_ylabel('Amplituda')
            self.noise_impact_ax_imag.set_title('Imaginarne komponente')
            self.noise_impact_ax_imag.grid(True)
            
            self.noise_impact_ax_real.legend()
            self.noise_impact_ax_imag.legend()
            self.noise_impact_canvas.draw()

        # Eye Diagram
        self.eye_diagram_ax_before.clear()
        self.eye_diagram_ax_after.clear()
        if self.received_symbols.size > 0:
            num_symbols_to_plot = min(self.received_symbols.shape[1], self.EYE_DIAGRAM_SYMBOLS)
            
            # Eye diagram before equalization
            num_rx_modes = int(self.num_rx_ant_entry.get()) * int(self.num_modes_entry.get())
            if self.received_symbols.size > 0:
                for mode_idx in range(num_rx_modes):
                    received_stream = self.received_symbols[mode_idx, :]
                    for i in range(0, len(received_stream) - 2, 1):
                        self.eye_diagram_ax_before.plot(np.real(received_stream[i:i + 2]), color='b', alpha=0.1)
                self.eye_diagram_ax_before.set_xlabel('Vrijeme (simboli)')
                self.eye_diagram_ax_before.set_ylabel('Amplituda')
                self.eye_diagram_ax_before.set_title('Eye Dijagram prije ekvalizacije')
                self.eye_diagram_ax_before.grid(True)

            # Eye diagram after equalization
            if self.equalized_symbols.size > 0:
                for mode_idx in range(num_rx_modes):
                    equalized_stream = self.equalized_symbols[mode_idx, :]
                    for i in range(0, len(equalized_stream) - 2, 1):
                        self.eye_diagram_ax_after.plot(np.real(equalized_stream[i:i + 2]), color='b', alpha=0.1)
                self.eye_diagram_ax_after.set_xlabel('Vrijeme (simboli)')
                self.eye_diagram_ax_after.set_ylabel('Amplituda')
                self.eye_diagram_ax_after.set_title('Eye Dijagram poslije ekvalizacije')
                self.eye_diagram_ax_after.grid(True)
        self.eye_diagram_canvas.draw()
        
        # Detaljni prikaz vlakna
        self.create_fiber_propagation_plot(fiber_length, attenuation)
        self.loading_label.pack_forget()
        self.show_all_plots()

    ## @brief Kreira i prikazuje graf propagacije signala kroz vlakno.
    ## @param fiber_length Dužina vlakna u km.
    ## @param attenuation Slabljenje vlakna u dB/km.
    ## @details Ova metoda kreira i prikazuje graf koji pokazuje snagu signala duž vlakna.
    def create_fiber_propagation_plot(self, fiber_length, attenuation):
        """
        @brief Kreira i prikazuje graf propagacije signala kroz vlakno.
        
        @param fiber_length Dužina vlakna u km.
        @param attenuation Slabljenje vlakna u dB/km.
        @details Ova metoda kreira i prikazuje graf koji pokazuje snagu signala duž vlakna.
        """
        ## @brief Kreira i prikazuje graf propagacije signala kroz vlakno.
        ## @param fiber_length Dužina vlakna u km.
        ## @param attenuation Slabljenje vlakna u dB/km.
        ## @details Ova metoda kreira i prikazuje graf koji pokazuje snagu signala duž vlakna.
        self.detailed_fiber_ax.clear()
        num_points = max(self.FIBER_PROPAGATION_POINTS, int(fiber_length))  # Ensure at least FIBER_PROPAGATION_POINTS or fiber_length points
        distance = np.linspace(0, fiber_length, num_points)
        # Calculate signal power in dBm
        signal_power_dbm = 10 * np.log10(1)  # Initial power is 1 mW (0 dBm)
        
        # Calculate signal power along the fiber
        signal_power_dbm_along_fiber = signal_power_dbm - attenuation * distance
        
        self.detailed_fiber_ax.plot(distance, signal_power_dbm_along_fiber, label='Signal Power (dBm)')
        self.detailed_fiber_ax.set_xlabel('Dužina vlakna (km)')
        self.detailed_fiber_ax.set_ylabel('Snaga signala (dBm)')
        self.detailed_fiber_ax.set_title(f'Prikaz vlakna (Dužina: {fiber_length} km, Slabljenje: {attenuation} dB/km)')
        self.detailed_fiber_ax.grid(True)
        self.detailed_fiber_ax.legend()
        self.detailed_fiber_canvas.draw()

    ## @brief Prikazuje popup prozor sa matricom kanala.
    ## @details Ova metoda kreira i prikazuje popup prozor koji sadrži unose matrice kanala.
    def show_channel_matrix_popup(self):
        """
        @brief Prikazuje popup prozor sa matricom kanala.
        
        @details Ova metoda kreira i prikazuje popup prozor koji sadrži unose matrice kanala.
        """
        ## @brief Prikazuje popup prozor sa matricom kanala.
        ## @details Ova metoda kreira i prikazuje popup prozor koji sadrži unose matrice kanala.
        if self.channel_matrix_popup is not None:
            self.channel_matrix_popup.destroy()
        
        self.channel_matrix_popup = Toplevel(self.master)
        self.channel_matrix_popup.title("Kanalna matrica (H)")
        
        matrix_frame = ttk.Frame(self.channel_matrix_popup)
        matrix_frame.pack(padx=10, pady=10)
        
        num_rows = int(self.num_rx_ant_entry.get()) * int(self.num_modes_entry.get())
        num_cols = int(self.num_tx_ant_entry.get()) * int(self.num_modes_entry.get())
        
        default_matrix = np.eye(min(num_rows, num_cols))
        
        if num_rows > num_cols:
            padding_rows = num_rows - num_cols
            padding = np.zeros((padding_rows, num_cols))
            default_matrix = np.vstack((default_matrix, padding))
        elif num_cols > num_rows:
            padding_cols = num_cols - num_rows
            padding = np.zeros((num_rows, padding_cols))
            default_matrix = np.hstack((default_matrix, padding))
        
        self.create_channel_matrix_entries(default_matrix.tolist(), matrix_frame, self.channel_matrix_popup)

    ## @brief Kreira i prikazuje unose matrice kanala u popup prozoru.
    ## @param matrix Matrica kanala za prikaz.
    ## @param matrix_frame Okvir u kojem se prikazuju unosi matrice.
    ## @param popup_window Prozor u kojem se prikazuju unosi matrice.
    ## @details Ova metoda kreira i prikazuje unose matrice kanala u popup prozoru, na osnovu date matrice i okvira.
    def create_channel_matrix_entries(self, matrix, matrix_frame, popup_window):
        """
        @brief Kreira i prikazuje unose matrice kanala u popup prozoru.
        
        @param matrix Matrica kanala za prikaz.
        @param matrix_frame Okvir u kojem se prikazuju unosi matrice.
        @param popup_window Prozor u kojem se prikazuju unosi matrice.
        @details Ova metoda kreira i prikazuje unose matrice kanala u popup prozoru, na osnovu date matrice i okvira.
        """
        ## @brief Kreira i prikazuje unose matrice kanala u popup prozoru.
        ## @param matrix Matrica kanala za prikaz.
        ## @param matrix_frame Okvir u kojem se prikazuju unosi matrice.
        ## @param popup_window Prozor u kojem se prikazuju unosi matrice.
        ## @details Ova metoda kreira i prikazuje unose matrice kanala u popup prozoru, na osnovu date matrice i okvira.
        # Clear existing entries
        if hasattr(popup_window, 'channel_matrix_entries'):
            for entry_row in popup_window.channel_matrix_entries:
                for entry in entry_row:
                    entry.destroy()
            popup_window.channel_matrix_entries = []

        # Create new entries
        if matrix_frame is not None:
            for i, row in enumerate(matrix):
                entry_row = []
                for j, value in enumerate(row):
                    entry = ttk.Label(matrix_frame, text=f"{abs(value):.2f} ∠ {np.angle(value):.2f}", width=10)
                    entry.complex_value = value
                    entry.grid(row=i, column=j, padx=1, pady=1)
                    entry_row.append(entry)
                popup_window.channel_matrix_entries.append(entry_row)

    ## @brief Preuzima matricu kanala iz GUI unosa.
    ## @return Matrica kanala kao NumPy niz.
    ## @details Ova metoda preuzima matricu kanala iz GUI unosa i vraća je kao NumPy niz.
    def get_channel_matrix_from_entries(self):
        """
        @brief Preuzima matricu kanala iz GUI unosa.
        
        @return Matrica kanala kao NumPy niz.
        @details Ova metoda preuzima matricu kanala iz GUI unosa i vraća je kao NumPy niz.
        """
        ## @brief Preuzima matricu kanala iz GUI unosa.
        ## @return Matrica kanala kao NumPy niz.
        ## @details Ova metoda preuzima matricu kanala iz GUI unosa i vraća je kao NumPy niz.
        matrix = []
        if hasattr(self.channel_matrix_popup, 'channel_matrix_entries'):
            for entry_row in self.channel_matrix_popup.channel_matrix_entries:
                row = []
                for entry in entry_row:
                    try:
                        text = entry.cget("text")
                        parts = text.split("∠")
                        mag = float(parts[0].strip())
                        phase = float(parts[1].strip())
                        row.append(complex(mag * np.cos(phase), mag * np.sin(phase)))
                    except:
                        row.append(0)
                matrix.append(row)
        return np.array(matrix)

    ## @brief Ažurira stanje unosa matrice kanala (samo za čitanje ili uređivanje).
    ## @details Ova metoda ažurira stanje unosa matrice kanala na osnovu zastavice `channel_matrix_entry_readonly`. Ako je `channel_matrix_entry_readonly` True, unosi su postavljeni samo za čitanje, inače su postavljeni za uređivanje.
    def update_channel_matrix_entry_state(self):
        """
        @brief Ažurira stanje unosa matrice kanala (samo za čitanje ili uređivanje).
        
        @details Ova metoda ažurira stanje unosa matrice kanala na osnovu zastavice `channel_matrix_entry_readonly`. Ako je `channel_matrix_entry_readonly` True, unosi su postavljeni samo za čitanje, inače su postavljeni za uređivanje.
        """
        ## @brief Ažurira stanje unosa matrice kanala (samo za čitanje ili uređivanje).
        ## @details Ova metoda ažurira stanje unosa matrice kanala na osnovu zastavice `channel_matrix_entry_readonly`. Ako je `channel_matrix_entry_readonly` True, unosi su postavljeni samo za čitanje, inače su postavljeni za uređivanje.
        for entry_row in self.channel_matrix_entries:
            for entry in entry_row:
                entry.config(state=tk.DISABLED if self.channel_matrix_entry_readonly else tk.NORMAL)
                
if __name__ == "__main__":
    root = tk.Tk()
    gui = QPSK_MIMO_GUI(root)
    root.mainloop()

