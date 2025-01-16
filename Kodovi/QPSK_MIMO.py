import tkinter as tk
from tkinter import ttk, Toplevel, Label, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy import signal
from numpy.linalg import svd, det, matrix_rank, inv

# Constants
MIN_NUM_BITS = 100
MAX_NUM_BITS = 10000
MIN_SNR_DB = 0
MAX_SNR_DB = 30
MIN_NUM_ANTENNAS = 1
MAX_NUM_ANTENNAS = 4
MIN_NUM_MODES = 1
MAX_NUM_MODES = 4
MIN_FIBER_LENGTH = 1
MAX_FIBER_LENGTH = 1000
MIN_ATTENUATION = 0.1
MAX_ATTENUATION = 1

## @brief This class implements a tooltip for Tkinter widgets.
class ToolTip:
    ## @brief Konstruktor za klasu ToolTip.
    ## @param widget Widget na koji se prikači tooltip.
    ## @param text Tekst koji se prikazuje u tooltipu.
    ## @details Inicijalizira tooltip sa datim widgetom i tekstom, te povezuje događaje prikaza i sakrivanja.
    def __init__(self, widget, text):
        """
        @brief Inicijalizira tooltip.
        
        @param widget Tkinter widget na koji se tooltip prikači.
        @param text Tekst koji se prikazuje u tooltipu.
        """
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    ## @brief Prikazuje tooltip prozor.
    ## @param event Događaj koji je pokrenuo tooltip.
    ## @details Ova metoda kreira i prikazuje tooltip prozor na trenutnoj poziciji miša.
    def show_tooltip(self, event=None):
        """
        @brief Prikazuje tooltip prozor.
        
        @param event Događaj koji je pokrenuo tooltip.
        @details Ova metoda kreira i prikazuje tooltip prozor na trenutnoj poziciji miša.
        """
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = Label(self.tooltip_window, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, padx=1, pady=1)
        label.pack()

    ## @brief Sakriva tooltip prozor.
    ## @param event Događaj koji je pokrenuo sakrivanje tooltippa.
    ## @details Ova metoda uništava tooltip prozor kada miš napusti widget.
    def hide_tooltip(self, event=None):
        """
        @brief Sakriva tooltip prozor.
        
        @param event Događaj koji je pokrenuo sakrivanje tooltippa.
        @details Ova metoda uništava tooltip prozor kada miš napusti widget.
        """
        if self.tooltip_window:
            self.tooltip_window.destroy()

## @brief This class implements the GUI for the QPSK MIMO simulation.
class QPSK_MIMO_GUI:
    ## @brief Konstruktor za klasu QPSK_MIMO_GUI.
    ## @param master Roditeljski prozor.
    ## @details Inicijalizira glavni prozor i sve GUI elemente za QPSK MIMO simulaciju.
    def __init__(self, master):
        """
        @brief Konstruktor za klasu QPSK_MIMO_GUI.
        
        @param master Roditeljski prozor.
        @details Inicijalizira glavni prozor i sve GUI elemente za QPSK MIMO simulaciju.
        """
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
        self.num_bits_label = ttk.Label(self.input_frame, text="Broj bita (100-10000):")
        self.num_bits_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.num_bits_label, "Ukupan broj generisanih bitova za simulaciju.")
        self.num_bits_entry = ttk.Entry(self.input_frame)
        self.num_bits_entry.insert(0, "1000")
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

        # Matrica kanala
        self.channel_label = ttk.Label(self.input_frame, text="Matrica kanala (H):")
        self.channel_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.channel_label, "Matrica koja opisuje propagaciju signala između predajnih i prijemnih antena/modova.")
        self.channel_matrix_button = ttk.Button(self.input_frame, text="Info", command=self.show_channel_matrix_popup)
        self.channel_matrix_button.grid(row=5, column=1, padx=5, pady=5)
        self.channel_matrix_entries = []
        self.channel_matrix_entry_readonly = True

        # Duljina vlakna (km)
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

        self.explain_button = ttk.Button(master, text="Objasni koncept", command=self.explain_concept)
        self.explain_button.pack(pady=5, side=tk.RIGHT, padx=10, anchor=tk.NE)

        self.help_button = ttk.Button(master, text="Pomoć", command=self.show_help)
        self.help_button.pack(pady=5, side=tk.RIGHT, padx=10, anchor=tk.NE)

        self.simulate_button = ttk.Button(master, text="Simuliraj", command=self.start_simulation)
        self.simulate_button.pack(pady=5, side=tk.LEFT, padx=10, anchor=tk.NW)

        self.reset_button = ttk.Button(master, text="Resetuj", command=self.reset_simulation)
        self.reset_button.pack(pady=5, side=tk.LEFT, padx=10, anchor=tk.NW)

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
        self.notebook.add(self.channel_tab, text="Matrica kanala")
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
        self.received_symbols = np.array([])
        self.equalized_symbols = np.array([])
        self.update_channel_matrix_size()
        self.channel_matrix_entries = []
        self.channel_matrix_entry_readonly = True
        self.channel_matrix_popup = None

    ## @brief Pokreće proces simulacije.
    ## @details Ova metoda inicira simulaciju postavljanjem unosa matrice kanala na uređivanje i pozivanjem metode simulacije.
    def start_simulation(self):
        """
        @brief Pokreće proces simulacije.
        
        @details Ova metoda pokreće simulaciju postavljanjem unosa matrice kanala na uređivanje i pozivanjem metode simulacije.
        """
        self.channel_matrix_entry_readonly = False
        self.simulate()

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

        H = np.zeros((num_rx_modes, num_tx_modes), dtype=complex)

        # More realistic mode coupling matrix
        coupling_matrix = np.zeros((num_rx_modes, num_tx_modes), dtype=complex)
        for i in range(num_rx_modes):
            for j in range(num_tx_modes):
                coupling = coupling_coeff * np.exp(-((i - j) / 2)**2)  # Gaussian-like coupling
                coupling_matrix[i, j] = coupling * (np.random.randn() + 1j * np.random.randn())
        
        # Ensure the diagonal is 1
        np.fill_diagonal(coupling_matrix, 1)

        # Differential mode delay (DMD) as a time delay
        dmd_matrix = np.zeros((num_rx_modes, num_tx_modes), dtype=complex)
        for i in range(num_rx_modes):
            for j in range(num_tx_modes):
                dmd_amount = dmd_coeff * (i - j)**2
                # Apply a frequency-dependent phase shift
                dmd_matrix[i, j] = np.exp(-1j * 2 * np.pi * dmd_amount)

        H = np.dot(dmd_matrix, coupling_matrix)
        return H

    ## @brief Prikazuje prozor za pomoć sa uputama o korištenju simulacije.
    ## @details Ova metoda kreira i prikazuje prozor za pomoć sa uputama o korištenju simulacije.
    def show_help(self):
        """
        @brief Prikazuje prozor za pomoć sa uputama o korištenju simulacije.
        
        @details Ova metoda kreira i prikazuje prozor za pomoć sa uputama o korištenju simulacije.
        """
        help_text = """
        Ovaj program simulira QPSK MIMO sistem zasnovan na višemodnom optičkom vlaknu.

        **Parametri simulacije:**
        - **Broj bita:** Ukupan broj generisanih bitova za simulaciju.
        - **SNR (dB):** Omjer signala i šuma u decibelima.
        - **Broj predajnih antena:** Broj predajnih antena u MIMO sistemu.
        - **Broj prijemnih antena:** Broj prijemnih antena u MIMO sistemu.
        - **Broj modova:** Broj prostornih modova u vlaknu.
        - **Matrica kanala (H):** Matrica koja opisuje propagaciju signala između predajnih i prijemnih antena/modova.
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

    ## @brief Prikazuje prozor koji objašnjava koncept QPSK MIMO modulacije u višemodnom vlaknu.
    ## @details Ova metoda kreira i prikazuje prozor sa detaljnim objašnjenjem QPSK MIMO koncepta u višemodnom vlaknu.
    def explain_concept(self):
        """
        @brief Prikazuje prozor koji objašnjava koncept QPSK MIMO modulacije u višemodnom vlaknu.
        
        @details Ova metoda kreira i prikazuje prozor sa detaljnim objašnjenjem QPSK MIMO koncepta u višemodnom vlaknu.
        """
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

    ## @brief Prikazuje prozor koji objašnjava koncept QPSK MIMO modulacije u višemodnom vlaknu.
    ## @details Ova metoda kreira i prikazuje prozor sa detaljnim objašnjenjem QPSK MIMO koncepta u višemodnom vlaknu.
    def explain_concept(self):
        """
        @brief Prikazuje prozor koji objašnjava koncept QPSK MIMO modulacije u višemodnom vlaknu.
        
        @details Ova metoda kreira i prikazuje prozor sa detaljnim objašnjenjem QPSK MIMO koncepta u višemodnom vlaknu.
        """
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
        self.snr_capacity_ax.clear()
        self.snr_capacity_canvas.draw()
        self.noise_impact_ax.clear()
        self.noise_impact_canvas.draw()
        self.detailed_fiber_ax.clear()
        self.detailed_fiber_canvas.draw()
        self.eye_diagram_ax_before.clear()
        self.eye_diagram_ax_after.clear()
        if self.fiber_propagation_ax:
            self.fiber_propagation_ax.clear()
            self.fiber_propagation_canvas.draw()
        self.channel_matrix_entry_readonly = True
        self.update_channel_matrix_entry_state()


        # Reset result labels
        self.ber_label_text.set("BER: N/A")
        self.snr_result_label_text.set("SNR (dB): N/A")
        self.capacity_label_text.set("Kapacitet (bps/Hz): N/A")
        self.channel_matrix_displayed = False

    ## @brief Resetuje simulaciju brisanjem svih grafova i rezultata.
    ## @details Ova metoda briše sve grafove i resetuje labele rezultata na njihova početna stanja.
    def reset_simulation(self):
        """
        @brief Resetuje simulaciju brisanjem svih grafova i rezultata.
        
        @details Ova metoda briše sve grafove i resetuje labele rezultata na njihova početna stanja.
        """
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
        self.snr_capacity_ax.clear()
        self.snr_capacity_canvas.draw()
        self.noise_impact_ax.clear()
        self.noise_impact_canvas.draw()
        self.detailed_fiber_ax.clear()
        self.detailed_fiber_canvas.draw()
        self.eye_diagram_ax_before.clear()
        self.eye_diagram_ax_after.clear()
        if self.fiber_propagation_ax:
            self.fiber_propagation_ax.clear()
            self.fiber_propagation_canvas.draw()
        self.channel_matrix_entry_readonly = True
        self.update_channel_matrix_entry_state()


        # Reset result labels
        self.ber_label_text.set("BER: N/A")
        self.snr_result_label_text.set("SNR (dB): N/A")
        self.capacity_label_text.set("Kapacitet (bps/Hz): N/A")
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
            
            default_matrix = np.eye(min(new_matrix_size))
            
            if new_matrix_size[0] > new_matrix_size[1]:
                padding_rows = new_matrix_size[0] - new_matrix_size[1]
                padding = np.zeros((padding_rows, new_matrix_size[1]))
                default_matrix = np.vstack((default_matrix, padding))
            elif new_matrix_size[1] > new_matrix_size[0]:
                padding_cols = new_matrix_size[1] - new_matrix_size[0]
                padding = np.zeros((new_matrix_size[0], padding_cols))
                default_matrix = np.hstack((default_matrix, padding))
            
            self.create_channel_matrix_entries(default_matrix.tolist(), None)
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
        # Dohvati parametre simulacije iz GUI
        try:
            num_bits = int(self.num_bits_entry.get())
        except ValueError:
            messagebox.showerror("Greška", "Broj bita mora biti cijeli broj.")
            return
        if not MIN_NUM_BITS <= num_bits <= MAX_NUM_BITS:
            messagebox.showerror("Greška", f"Broj bita mora biti između {MIN_NUM_BITS} i {MAX_NUM_BITS}.")
            return

        try:
            snr_db = float(self.snr_entry.get())
        except ValueError:
             messagebox.showerror("Greška", "SNR mora biti broj.")
             return
        if not MIN_SNR_DB <= snr_db <= MAX_SNR_DB:
            messagebox.showerror("Greška", f"SNR mora biti između {MIN_SNR_DB} i {MAX_SNR_DB} dB.")
            return

        try:
            fiber_length = float(self.fiber_length_entry.get())
        except ValueError:
            messagebox.showerror("Greška", "Dužina vlakna mora biti broj.")
            return
        if not MIN_FIBER_LENGTH <= fiber_length <= MAX_FIBER_LENGTH:
            messagebox.showerror("Greška", f"Dužina vlakna mora biti između {MIN_FIBER_LENGTH} i {MAX_FIBER_LENGTH} km.")
            return

        try:
            attenuation = float(self.attenuation_entry.get())
        except ValueError:
            messagebox.showerror("Greška", "Koeficijent slabljenja mora biti broj.")
            return
        if not MIN_ATTENUATION <= attenuation <= MAX_ATTENUATION:
            messagebox.showerror("Greška", f"Koeficijent slabljenja mora biti između {MIN_ATTENUATION} i {MAX_ATTENUATION} dB/km.")
            return

        num_modes = int(self.num_modes_entry.get())
        if not 1 <= num_modes <= 4:
            messagebox.showerror("Greška", "Broj modova mora biti između 1 i 4.")
            return

        num_tx_antennas = int(self.num_tx_ant_entry.get())
        if not 1 <= num_tx_antennas <= 4:
            messagebox.showerror("Greška", "Broj predajnih antena mora biti između 1 i 4.")
            return
        num_rx_antennas = int(self.num_rx_ant_entry.get())
        if not 1 <= num_rx_antennas <= 4:
            messagebox.showerror("Greška", "Broj prijemnih antena mora biti između 1 i 4.")
            return
        
        self.channel_matrix_entry_readonly = True
        self.update_channel_matrix_entry_state()

        num_tx_modes = num_tx_antennas * num_modes
        num_rx_modes = num_rx_antennas * num_modes
        coupling_coeff = 0.01 * np.sqrt(fiber_length / 100)  # Non-linear scaling
        dmd_coeff = 0.001 * np.sqrt(fiber_length / 100) # Non-linear scaling
        
        H = self.get_channel_matrix_from_entries()
        if H is None:
            return
        if H.shape != (num_rx_modes, num_tx_modes):
            H = self.generate_channel_matrix(num_tx_modes, num_rx_modes, fiber_length, seed=42, coupling_coeff=coupling_coeff, dmd_coeff=dmd_coeff)

        bits = np.random.randint(0, 2, num_bits)

        # QPSK Modulacija
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

        # MIMO dio
        tx_signals = np.tile(qpsk_symbols, (num_tx_antennas * num_modes, 1))

        # Mode coupling matrix (more realistic)
        coupling_strength = 0.1 * (fiber_length / 100)  # Coupling strength increases with fiber length
        coupling_matrix = np.eye(num_tx_antennas * num_modes) + coupling_strength * np.random.randn(num_tx_antennas * num_modes, num_tx_antennas * num_modes)
        tx_signals_coupled = np.dot(coupling_matrix, tx_signals)

        # Modal dispersion (more realistic)
        num_symbols = tx_signals_coupled.shape[1]
        freq = np.fft.fftfreq(num_symbols)
        delayed_signals = np.zeros_like(tx_signals_coupled, dtype=complex)
        for mode_idx in range(num_tx_antennas * num_modes):
            mode_signal_fft = np.fft.fft(tx_signals_coupled[mode_idx, :])
            transfer_function = H[mode_idx, mode_idx]
            delayed_mode_signal = np.fft.ifft(mode_signal_fft * transfer_function)
            delayed_signals[mode_idx, :] = delayed_mode_signal

        # Prijem signala
        rx_signals = np.dot(H, delayed_signals)

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
        demodulated_bits = []
        for rx_signal in self.equalized_symbols.T:
            for symbol in rx_signal:
                min_dist = float('inf')
                closest_symbol = None
                for ref_symbol in mapping.values():
                    dist = np.abs(symbol - ref_symbol)**2
                    if dist < min_dist:
                        min_dist = dist
                        closest_symbol = ref_symbol
                if closest_symbol is not None:
                    demodulated_bits.extend(inverse_mapping[closest_symbol])

        # Proračun BER
        ber_point = np.nan
        if len(tx_bits) > 0:
            ber_point = np.sum(np.array(tx_bits) != np.array(demodulated_bits[:len(tx_bits)])) / len(tx_bits)
            self.ber_label_text.set(f"BER: {ber_point:.4f}")
        else:
            self.ber_label_text.set("BER: N/A")

        self.snr_result_label_text.set(f"SNR (dB): {snr_db:.2f}")

        # Proračun kapaciteta (koristeći SVD)
        if num_rx_antennas * num_modes > 0 and num_tx_antennas * num_modes > 0:
            U, s, V = svd(H)
            capacity_point = 0
            snr_linear = 10**(snr_db / 10)
            for singular_value in s:
                capacity_point += np.log2(1 + (snr_linear / (num_tx_antennas * num_modes)) * (singular_value**2))
            self.capacity_label_text.set(f"Kapacitet (bps/Hz): {capacity_point:.2f}")
        else:
            self.capacity_label_text.set("Kapacitet (bps/Hz): N/A")
        self.channel_matrix_entry_readonly = True

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
        
        for i in range(int(self.num_rx_ant_entry.get()) * int(self.num_modes_entry.get())):
            self.constellation_ax.plot(self.received_symbols[i].real, self.received_symbols[i].imag, '.', label=f'Primljeni simboli (Prijemnik {i+1})')


        self.constellation_ax.set_xlabel('In-phase')
        self.constellation_ax.set_ylabel('Quadrature')
        self.constellation_ax.set_title('Konstelacijski dijagram (QPSK MIMO)')
        
        # Calculate plot limits with a margin
        all_real = np.concatenate([self.received_symbols[i].real for i in range(int(self.num_rx_ant_entry.get()) * int(self.num_modes_entry.get()))])
        all_imag = np.concatenate([self.received_symbols[i].imag for i in range(int(self.num_rx_ant_entry.get()) * int(self.num_modes_entry.get()))])
        
        margin_x = 0.2 * (max(all_real) - min(all_real))
        margin_y = 0.25 * (max(all_imag) - min(all_imag))
        
        self.constellation_ax.set_xlim(min(all_real) - margin_x, max(all_real) + margin_x)
        self.constellation_ax.set_ylim(min(all_imag) - margin_y, max(all_imag) + margin_y)
        
        self.constellation_ax.grid(True)
        self.constellation_ax.legend()
        self.constellation_canvas.draw()
        self.constellation_figure.tight_layout(pad=3.0)

        # Prikaz matrice kanala
        if not self.channel_matrix_displayed:
            if num_rx_antennas * num_modes > 0 and num_tx_antennas * num_modes > 0:
                # Magnitude plot
                im_mag = self.channel_ax_mag.imshow(np.abs(H), cmap='viridis')
                self.channel_figure.colorbar(im_mag, ax=self.channel_ax_mag, fraction=0.046, pad=0.04, label='Amplituda')
                self.channel_ax_mag.set_xticks(np.arange(num_tx_antennas * num_modes))
                self.channel_ax_mag.set_yticks(np.arange(num_rx_antennas * num_modes))
                self.channel_ax_mag.set_xlabel('Predajni elementi')
                self.channel_ax_mag.set_ylabel('Prijemni elementi')
                self.channel_ax_mag.set_title('Magnituda matrice kanala (H)')
                self.channel_ax_mag.set_xlim(0, num_tx_antennas * num_modes - 1)
                self.channel_ax_mag.set_ylim(num_rx_antennas * num_modes - 1, 0)

                # Phase plot
                im_phase = self.channel_ax_phase.imshow(np.angle(H), cmap='twilight')
                self.channel_figure.colorbar(im_phase, ax=self.channel_ax_phase, fraction=0.046, pad=0.04, label='Faza (rad)')
                self.channel_ax_phase.set_xticks(np.arange(num_tx_antennas * num_modes))
                self.channel_ax_phase.set_yticks(np.arange(num_rx_antennas * num_modes))
                self.channel_ax_phase.set_xlabel('Predajni elementi')
                self.channel_ax_phase.set_ylabel('Prijemni elementi')
                self.channel_ax_phase.set_title('Faza matrice kanala (H)')
                self.channel_ax_phase.set_xlim(0, num_tx_antennas * num_modes - 1)
                self.channel_ax_phase.set_ylim(num_rx_antennas * num_modes - 1, 0)
            self.channel_canvas.draw()
            self.channel_matrix_displayed = True
        
        # SNR vs BER plot
        snr_db_range = np.linspace(0, 20, 10)
        ber_values = []
        for current_snr_db in snr_db_range:
            snr_linear = 10**(current_snr_db / 10)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power / 2)
            noise = noise_std * (np.random.randn(*rx_signals.shape) + 1j * np.random.randn(*rx_signals.shape))
            received_symbols_ber = rx_signals + noise
            demodulated_bits_ber = []
            for rx_signal in received_symbols_ber.T:
                for symbol in rx_signal:
                    min_dist = float('inf')
                    closest_symbol = None
                    for ref_symbol in mapping.values():
                        dist = np.abs(symbol - ref_symbol)**2
                        if dist < min_dist:
                            min_dist = dist
                            closest_symbol = ref_symbol
                    if closest_symbol is not None:
                        demodulated_bits_ber.extend(inverse_mapping[closest_symbol])
            ber = np.nan
            if len(tx_bits) > 0:
                ber = np.sum(np.array(tx_bits) != np.array(demodulated_bits_ber[:len(tx_bits)])) / len(tx_bits)
            ber_values.append(ber)

        self.snr_ber_ax.clear()
        self.snr_ber_ax.semilogy(snr_db_range, ber_values, marker='o', linestyle='-')
        self.snr_ber_ax.set_xlabel('SNR (dB)')
        self.snr_ber_ax.set_ylabel('BER')
        self.snr_ber_ax.set_title('Ovisnost BER o SNR')
        self.snr_ber_ax.grid(True)
        self.snr_ber_canvas.draw()

        # SNR vs Kapacitet plot
        capacity_values = []
        for current_snr_db in snr_db_range:
            snr_linear = 10**(current_snr_db / 10)
            if num_rx_antennas * num_modes > 0 and num_tx_antennas * num_modes > 0:
                capacity = np.log2(det(np.eye(num_rx_antennas * num_modes) + (snr_linear / (num_tx_antennas * num_modes)) * np.dot(H, H.conj().T)))
                capacity_values.append(capacity)
            else:
                capacity_values.append(np.nan)

        self.snr_capacity_ax.clear()
        self.snr_capacity_ax.plot(snr_db_range, capacity_values, marker='o', linestyle='-')
        self.snr_capacity_ax.set_xlabel('SNR (dB)')
        self.snr_capacity_ax.set_ylabel('Kapacitet (bps/Hz)')
        self.snr_capacity_ax.set_title('Ovisnost kapaciteta o SNR')
        self.snr_capacity_ax.grid(True)
        self.snr_capacity_canvas.draw()

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
            self.noise_impact_ax_real.legend()
            self.noise_impact_ax_real.grid(True)
            
            # Plot imaginary components
            self.noise_impact_ax_imag.plot(time, np.imag(tx_signals[0, :]), label='Odašiljani signal (Q)')
            self.noise_impact_ax_imag.plot(time, np.imag(noise[0, :]), label='Šum (Q)')
            self.noise_impact_ax_imag.plot(time, np.imag(self.received_symbols[0, :]), label='Primljeni signal (Q)')
            self.noise_impact_ax_imag.set_xlabel('Vrijeme (uzorci)')
            self.noise_impact_ax_imag.set_ylabel('Amplituda')
            self.noise_impact_ax_imag.set_title('Imaginarne komponente')
            self.noise_impact_ax_imag.legend()
            self.noise_impact_ax_imag.grid(True)
            
            self.noise_impact_canvas.draw()

        # Eye Diagram
        self.eye_diagram_ax_before.clear()
        self.eye_diagram_ax_after.clear()
        if self.received_symbols.size > 0:
            num_symbols_to_plot = min(self.received_symbols.shape[1], 2000)  # Increased for clarity
            
            # Eye diagram before equalization
            received_stream = self.received_symbols.flatten()
            for i in range(0, len(received_stream) - 2 * int(self.num_tx_ant_entry.get()) * int(self.num_modes_entry.get()), int(self.num_tx_ant_entry.get()) * int(self.num_modes_entry.get())):
                self.eye_diagram_ax_before.plot(np.real(received_stream[i:i + 2 * int(self.num_tx_ant_entry.get()) * int(self.num_modes_entry.get())]), color='b', alpha=0.1)
            self.eye_diagram_ax_before.set_xlabel('Vrijeme (simboli)')
            self.eye_diagram_ax_before.set_ylabel('Amplituda')
            self.eye_diagram_ax_before.set_title('Eye Dijagram prije ekvalizacije')
            self.eye_diagram_ax_before.grid(True)

            # Eye diagram after equalization
            if self.equalized_symbols.size > 0:
                equalized_stream = self.equalized_symbols.flatten()
                for i in range(0, len(equalized_stream) - 2 * int(self.num_tx_ant_entry.get()) * int(self.num_modes_entry.get()), int(self.num_tx_ant_entry.get()) * int(self.num_modes_entry.get())):
                    self.eye_diagram_ax_after.plot(np.real(equalized_stream[i:i + 2 * int(self.num_tx_ant_entry.get()) * int(self.num_modes_entry.get())]), color='b', alpha=0.1)
                self.eye_diagram_ax_after.set_xlabel('Vrijeme (simboli)')
                self.eye_diagram_ax_after.set_ylabel('Amplituda')
                self.eye_diagram_ax_after.set_title('Eye Dijagram poslije ekvalizacije')
                self.eye_diagram_ax_after.grid(True)
        self.eye_diagram_canvas.draw()

        # Detaljni prikaz vlakna
        self.create_fiber_propagation_plot(fiber_length, attenuation)

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
        self.detailed_fiber_ax.clear()
        distance = np.linspace(0, fiber_length, 100)
        attenuation_linear = 10**(-attenuation / 10)
        signal_power_along_fiber = np.exp(-attenuation_linear * distance) # Simplified model

        self.detailed_fiber_ax.plot(distance, signal_power_along_fiber, label='Snaga signala')
        self.detailed_fiber_ax.set_xlabel('Duljina vlakna (km)')
        self.detailed_fiber_ax.set_ylabel('Snaga signala (relativno)')
        self.detailed_fiber_ax.set_title(f'Prikaz vlakna (Duljina: {fiber_length} km, Atenuacija: {attenuation} dB/km)')
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
        if self.channel_matrix_popup is not None:
            self.channel_matrix_popup.destroy()
        
        if self.channel_matrix_popup is not None:
            self.channel_matrix_popup.destroy()
        
        self.channel_matrix_popup = Toplevel(self.master)
        self.channel_matrix_popup.title("Matrica kanala (H)")
        
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
        
        self.create_channel_matrix_entries(default_matrix.tolist(), matrix_frame)
        
        # Display matrix dimensions
        matrix_label = ttk.Label(matrix_frame, text=f"Dimenzije matrice: [{num_rows}, {num_cols}]")
        matrix_label.pack(pady=5)

    ## @brief Kreira i prikazuje unose matrice kanala u popup prozoru.
    ## @param matrix Matrica kanala za prikaz.
    ## @param matrix_frame Okvir u kojem se prikazuju unosi matrice.
    ## @details Ova metoda kreira i prikazuje unose matrice kanala u popup prozoru, na osnovu date matrice i okvira.
    def create_channel_matrix_entries(self, matrix, matrix_frame):
        """
        @brief Kreira i prikazuje unose matrice kanala u popup prozoru.
        
        @param matrix Matrica kanala za prikaz.
        @param matrix_frame Okvir u kojem se prikazuju unosi matrice.
        @details Ova metoda kreira i prikazuje unose matrice kanala u popup prozoru, na osnovu date matrice i okvira.
        """
        # Clear existing entries
        for entry_row in self.channel_matrix_entries:
            for entry in entry_row:
                entry.destroy()
        self.channel_matrix_entries = []

        # Create new entries
        if matrix_frame is not None:
            for i, row in enumerate(matrix):
                entry_row = []
                for j, value in enumerate(row):
                    entry = ttk.Label(matrix_frame, text=str(int(value)), width=5)
                    entry.grid(row=i, column=j, padx=1, pady=1)
                    entry_row.append(entry)
                self.channel_matrix_entries.append(entry_row)

    ## @brief Preuzima matricu kanala iz GUI unosa.
    ## @return Matrica kanala kao NumPy niz, ili None ako dođe do greške.
    ## @details Ova metoda preuzima matricu kanala iz GUI unosa i vraća je kao NumPy niz.
    def get_channel_matrix_from_entries(self):
        """
        @brief Preuzima matricu kanala iz GUI unosa.
        
        @return Matrica kanala kao NumPy niz, ili None ako dođe do greške.
        @details Ova metoda preuzima matricu kanala iz GUI unosa i vraća je kao NumPy niz.
        """
        matrix = []
        for entry_row in self.channel_matrix_entries:
            row = []
            for entry in entry_row:
                try:
                    row.append(complex(entry.cget("text")))
                except ValueError:
                    messagebox.showerror("Greška", "Neispravan unos u matrici kanala.")
                    return None
            matrix.append(row)
        return np.array(matrix)

    ## @brief Ažurira stanje unosa matrice kanala (samo za čitanje ili uređivanje).
    ## @details Ova metoda ažurira stanje unosa matrice kanala na osnovu zastavice `channel_matrix_entry_readonly`. Ako je `channel_matrix_entry_readonly` True, unosi su postavljeni samo za čitanje, inače su postavljeni za uređivanje.
    def update_channel_matrix_entry_state(self):
        """
        @brief Ažurira stanje unosa matrice kanala (samo za čitanje ili uređivanje).
        
        @details Ova metoda ažurira stanje unosa matrice kanala na osnovu zastavice `channel_matrix_entry_readonly`. Ako je `channel_matrix_entry_readonly` True, unosi su postavljeni samo za čitanje, inače su postavljeni za uređivanje.
        """
        if self.channel_matrix_entry_readonly:
            for entry_row in self.channel_matrix_entries:
                for entry in entry_row:
                    entry.config(state=tk.DISABLED)
        else:
            for entry_row in self.channel_matrix_entries:
                for entry in entry_row:
                    entry.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    gui = QPSK_MIMO_GUI(root)
    root.mainloop()

