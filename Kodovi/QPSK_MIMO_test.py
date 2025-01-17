import unittest
from QPSK_MIMO import QPSK_MIMO_GUI
import matplotlib.pyplot as plt
import numpy as np

class TestQPSK_MIMO(unittest.TestCase):

    def setUp(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.gui = QPSK_MIMO_GUI(self.root)

    ## @brief Testira BER kada nema grešaka u prijemnim bitima.
    ## @details Testira slučaj kada su svi primljeni biti identični poslanim.
    def test_ber_bez_gresaka(self):
        """
        @brief Testira BER kada nema grešaka u prijemnim bitima.
        @details Testira slučaj kada su svi primljeni biti identični poslanim.
        """
        print("\nTest: BER bez grešaka")
        tx_bits = [0, 1, 0, 1]
        demodulated_bits = [0, 1, 0, 1]
        ber = self.gui.calculate_ber(tx_bits, demodulated_bits)
        print(f"  Predajni biti: {tx_bits}")
        print(f"  Primljeni biti: {demodulated_bits}")
        print(f"  Izračunati BER: {ber}")
        self.assertEqual(ber, 0)

    ## @brief Testira BER kada su svi prijemni biti pogrešni.
    ## @details Testira slučaj kada su svi primljeni biti suprotni od poslanih.
    def test_ber_all_errors(self):
        """
        @brief Testira BER kada su svi prijemni biti pogrešni.
        @details Testira slučaj kada su svi primljeni biti suprotni od poslanih.
        """
        print("\nTest: BER sa svim greškama")
        tx_bits = [0, 1, 0, 1]
        demodulated_bits = [1, 0, 1, 0]
        ber = self.gui.calculate_ber(tx_bits, demodulated_bits)
        print(f"  Predajni biti: {tx_bits}")
        print(f"  Primljeni biti: {demodulated_bits}")
        print(f"  Izračunati BER: {ber}")
        self.assertEqual(ber, 1)

    ## @brief Testira BER kada je broj demoduliranih bita manji od broja poslanih bita.
    ## @details Testira slučaj kada je broj primljenih bita manji od broja poslanih bita.
    def test_ber_less_demodulated_bits(self):
        """
        @brief Testira BER kada je broj demoduliranih bita manji od broja poslanih bita.
        @details Testira slučaj kada je broj primljenih bita manji od broja poslanih bita.
        """
        print("\nTest: BER sa manje demoduliranih bita")
        tx_bits = [0, 1, 0, 1]
        demodulated_bits = [0, 1]
        ber = self.gui.calculate_ber(tx_bits, demodulated_bits)
        print(f"  Predajni biti: {tx_bits}")
        print(f"  Primljeni biti: {demodulated_bits}")
        print(f"  Izračunati BER: {ber}")
        self.assertEqual(ber, 0)

    ## @brief Testira BER kada su predajni biti prazni.
    ## @details Testira slučaj kada nema poslanih bita.
    def test_ber_empty_tx_bits(self):
        """
        @brief Testira BER kada su predajni biti prazni.
        @details Testira slučaj kada nema poslanih bita.
        """
        print("\nTest: BER sa praznim predajnim bitima")
        tx_bits = []
        demodulated_bits = [0, 1, 0, 1]
        ber = self.gui.calculate_ber(tx_bits, demodulated_bits)
        print(f"  Predajni biti: {tx_bits}")
        print(f"  Primljeni biti: {demodulated_bits}")
        print(f"  Izračunati BER: {ber}")
        self.assertTrue(np.isnan(ber))  # Check for NaN

    ## @brief Testira BER kada su prijemni biti prazni.
    ## @details Testira slučaj kada nema primljenih bita.
    def test_ber_empty_demodulated_bits(self):
        """
        @brief Testira BER kada su prijemni biti prazni.
        @details Testira slučaj kada nema primljenih bita.
        """
        print("\nTest: BER sa praznim prijemnim bitima")
        tx_bits = [0, 1, 0, 1]
        demodulated_bits = []
        ber = self.gui.calculate_ber(tx_bits, demodulated_bits)
        print(f"  Predajni biti: {tx_bits}")
        print(f"  Primljeni biti: {demodulated_bits}")
        print(f"  Izračunati BER: {ber}")
        self.assertEqual(ber, 0)

    ## @brief Testira BER kada su dužine predajnih i prijemnih bita nejednake.
    ## @details Testira slučaj kada broj poslanih i primljenih bita nije isti.
    def test_ber_unequal_lengths(self):
        """
        @brief Testira BER kada su dužine predajnih i prijemnih bita nejednake.
        @details Testira slučaj kada broj poslanih i primljenih bita nije isti.
        """
        print("\nTest: BER sa nejednakim dužinama bita")
        tx_bits = [0, 1, 0, 1, 0, 1]
        demodulated_bits = [0, 1, 1, 0]
        ber = self.gui.calculate_ber(tx_bits, demodulated_bits)
        expected_ber = 2/6
        print(f"  Predajni biti: {tx_bits}")
        print(f"  Primljeni biti: {demodulated_bits}")
        print(f"  Izračunati BER: {ber}")
        self.assertEqual(ber, expected_ber)

    ## @brief Testira BER kada postoje neke greške u prijemnim bitima.
    ## @details Testira slučaj kada su neki primljeni biti različiti od poslanih.
    def test_ber_with_some_errors(self):
        """
        @brief Testira BER kada postoje neke greške u prijemnim bitima.
        @details Testira slučaj kada su neki primljeni biti različiti od poslanih.
        """
        print("\nTest: BER sa nekim greškama")
        tx_bits = [0, 1, 0, 1, 0, 1]
        demodulated_bits = [0, 0, 1, 1, 0, 1]
        ber = self.gui.calculate_ber(tx_bits, demodulated_bits)
        expected_ber = 2/6
        print(f"  Predajni biti: {tx_bits}")
        print(f"  Primljeni biti: {demodulated_bits}")
        print(f"  Izračunati BER: {ber}")
        self.assertEqual(ber, expected_ber)

    ## @brief Testira BER sa dugim nizovima bita i svim greškama.
    ## @details Testira slučaj kada su svi primljeni biti suprotni od poslanih u dugom nizu.
    def test_ber_long_sequences(self):
        """
        @brief Testira BER sa dugim nizovima bita i svim greškama.
        @details Testira slučaj kada su svi primljeni biti suprotni od poslanih u dugom nizu.
        """
        print("\nTest: BER sa dugim nizovima bita i svim greškama")
        tx_bits = [0, 1] * 500
        demodulated_bits = [1, 0] * 500
        ber = self.gui.calculate_ber(tx_bits, demodulated_bits)
        print(f"  Predajni biti (prvih 10): {tx_bits[:10]}...")
        print(f"  Primljeni biti (prvih 10): {demodulated_bits[:10]}...")
        print(f"  Izračunati BER: {ber}")
        self.assertEqual(ber, 1)

    ## @brief Testira BER sa dugim nizovima bita bez grešaka.
    ## @details Testira slučaj kada su svi primljeni biti identični poslanim u dugom nizu.
    def test_ber_long_sequences_no_errors(self):
        """
        @brief Testira BER sa dugim nizovima bita bez grešaka.
        @details Testira slučaj kada su svi primljeni biti identični poslanim u dugom nizu.
        """
        print("\nTest: BER sa dugim nizovima bita bez grešaka")
        tx_bits = [0, 1] * 500
        demodulated_bits = [0, 1] * 500
        ber = self.gui.calculate_ber(tx_bits, demodulated_bits)
        print(f"  Predajni biti (prvih 10): {tx_bits[:10]}...")
        print(f"  Primljeni biti (prvih 10): {demodulated_bits[:10]}...")
        print(f"  Izračunati BER: {ber}")
        self.assertEqual(ber, 0)

    def tearDown(self):
        self.root.destroy()
        plt.close('all')

if __name__ == '__main__':
    unittest.main()
