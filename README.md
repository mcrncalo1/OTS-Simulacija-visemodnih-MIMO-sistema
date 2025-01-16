# Optički telekomunikacijski sistemi - Simulacija i analiza MIMO sistema zasnovanih na vlaknima s više modova

## Uvod

Ovaj projekat se bavi simulacijom i analizom MIMO (Multiple-Input Multiple-Output) sistema u optičkim telekomunikacijama, s posebnim fokusom na primjenu u vlaknima s više modova (FMF). Cilj je istražiti kako se MIMO tehnologije mogu koristiti za povećanje kapaciteta i pouzdanosti optičkih komunikacijskih sistema.

## Ključni Pojmovi

### Gubici u Optičkim Vlaknima

*   **Slabljenje (Attenuation):** Gubitak snage signala tokom prijenosa kroz optičko vlakno, mjeren u decibelima (dB).
*   **Apsorpcija materijala:** Gubitak optičke energije uslijed apsorpcije svjetlosti od strane materijala vlakna.
*   **Gubici uslijed raspršivanja:** Gubitak snage signala uslijed raspršivanja svjetlosti izvan jezgre vlakna.
*   **Nelinearni gubici:** Gubici koji se javljaju pri visokim nivoima optičke snage, uključujući stimulirano Ramanovo raspršivanje.
*   **Gubici uslijed savijanja:** Gubitak snage signala uslijed savijanja optičkog vlakna.
*   **Gubici uslijed spajanja modova:** Gubitak snage uslijed prijenosa energije između različitih modova u vlaknu.

### MIMO Sistemi

*   **MIMO (Multiple-Input Multiple-Output):** Tehnologija koja koristi više antena za prijenos i prijem signala, povećavajući kapacitet i pouzdanost komunikacije.
*   **BER (Bit Error Rate):** Stopa grešaka u bitovima, mjera kvalitete prijenosa podataka.
*   **Prostorno multipleksiranje (Spatial Multiplexing):** Tehnika koja omogućava istovremeni prijenos više tokova podataka putem različitih antena.
*   **SDM (Spatial Division Multiplexing):** Metoda za povećanje kapaciteta prijenosa korištenjem više prostornih modova u vlaknu.
*   **FMF (Few-Mode Fiber):** Optičko vlakno koje podržava prijenos nekoliko prostornih modova.
*   **λ-MIMO:** Napredna varijanta MIMO tehnologije koja koristi različite valne duljine (λ) za prijenos više signala kroz jedno optičko vlakno.
*   **Massive MIMO:** MIMO sistem s velikim brojem antena, ključan za buduće širokopojasne bežične mreže.

### Obrada Signala i Mreže

*   **MMSE (Minimum Mean Square Error):** Algoritam za izjednačavanje signala koji minimizira srednju kvadratnu grešku.
*   **OFDM-PON (Orthogonal Frequency Division Multiplexing Passive Optical Network):** Tehnologija za prijenos podataka putem optičke mreže koja koristi OFDM modulaciju.
*   **RoF (Radio over Fiber):** Tehnika prijenosa radio signala putem optičkog vlakna.
*   **Hromatska disperzija (Chromatic Dispersion):** Širenje optičkih impulsa tokom prijenosa kroz vlakno, ograničava brzinu prijenosa podataka.
*   **Digitalna obrada signala (DSP):** Obrada signala u digitalnom obliku, koristi se za poboljšanje kvalitete signala i kompenzaciju raznih efekata u komunikacijskom sistemu.

## Simulacija QPSK MIMO sistema

Ovaj Python kod implementira grafičko sučelje (GUI) za simulaciju QPSK (Quadrature Phase-Shift Keying) MIMO (Multiple-Input Multiple-Output) sistema u višemodnom optičkom vlaknu. Koristi biblioteke `tkinter` za GUI, `numpy` za numeričke operacije, `matplotlib` za vizualizaciju i `scipy` za obradu signala.

### Funkcionalnost

1.  **GUI Parametri:**
    *   Korisnik može postaviti parametre simulacije kao što su broj bita, SNR (omjer signala i šuma), broj predajnih i prijemnih antena, broj modova, dužinu vlakna i koeficijent slabljenja.
    *   Matrica kanala se može postaviti ručno ili generirati automatski.
2.  **Simulacija:**
    *   Generiše se niz bitova koji se zatim moduliraju korištenjem QPSK modulacije.
    *   Simulira se prijenos signala kroz MIMO kanal, uključujući efekte sprezanja modova i disperzije.
    *   Dodaje se AWGN (Additive White Gaussian Noise) šum na primljeni signal.
    *   Primjenjuje se MMSE (Minimum Mean Square Error) ekvalizator za poboljšanje kvalitete signala.
    *   Demoduliraju se primljeni simboli i izračunava se BER (Bit Error Rate).
3.  **Vizualizacija:**
    *   Prikazuje se konstelacijski dijagram odašiljanih i primljenih signala.
    *   Prikazuje se matrica kanala (magnituda i faza).
    *   Prikazuje se eye dijagram primljenog signala prije i poslije ekvalizacije.
    *   Prikazuje se utjecaj šuma na signal u vremenskoj domeni.
    *   Prikazuje se ovisnost BER o SNR i kapaciteta o SNR.
    *   Prikazuje se slabljenje signala duž vlakna.

### Biblioteke

*   `tkinter`: Za izradu grafičkog korisničkog sučelja.
*   `numpy`: Za numeričke operacije i manipulaciju matricama.
*   `matplotlib`: Za vizualizaciju podataka i crtanje grafova.
*   `scipy`: Za obradu signala i napredne matematičke funkcije.

### Kako koristiti

Za pokretanje simulacije, potrebno je pokrenuti skriptu `QPSK_MIMO.py`. Nakon pokretanja, korisnik može unijeti željene parametre simulacije u grafičkom sučelju (GUI). Simulacija se pokreće klikom na dugme "Simuliraj". Rezultati simulacije i grafovi se prikazuju u odgovarajućim tabovima. Za brisanje svih grafova i rezultata, koristi se dugme "Resetuj". Dodatne upute o korištenju simulacije mogu se pronaći klikom na dugme "Pomoć", a detaljnije objašnjenje QPSK MIMO koncepta u višemodnom vlaknu dostupno je klikom na dugme "Objasni koncept".
