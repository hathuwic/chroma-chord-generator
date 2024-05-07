# CHROMA CHORD GENERATOR
# --------------------------
# This script is the core of the real-time chord generation.
# It expects melody chroma values as integers over OSC on port 10000,
# and returns chords as list[[pitch: int, velocity: int]] over OSC on port 11000
# --------------------------
# Last updated: 14 May 2024

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Gets TensorFlow to shut up...

from chord_generation_utils.chord_generator import ChordGenerator
from chord_generation_utils.osc import OSCHandler

def main() -> None:
    
    try:

        if len(sys.argv) != 1:
            if len(sys.argv) != 4:
                print("Expected either no arguments, or 3 arguments: <ip> <receive_port> <send_port>")
                return
            ip = str(sys.argv[1])
            server_port = int(sys.argv[2])
            client_port = int(sys.argv[3])
        else:
            ip = "127.0.0.1"
            server_port = 10000
            client_port = 11000

        
        print("------------------------------")
        print("--- CHROMA-CHORD-GENERATOR ---")
        print("------------------------------")
        print("---- Press ctrl+c to exit ----")
        print("------------------------------")

        # Initialise the chord generator object
        # which houses the model and infrastructure
        chord_generator = ChordGenerator(
            model_path="./src/trained_model/chroma_histogram_generator_model",
            sequence_length=8,
            tonic=0, 
            chord_note_threshold=0.14,
            threshold_input_sequence=True,
            update_direction="append"
        )

        # Initialise the OSC handler object which communicates
        # with Max, receiving melody note numbers and returning chords
        # as [pitch, velocity] pairs
        osc_handler = OSCHandler(
            chord_generator,
            ip=ip,
            server_port=server_port,
            client_port=client_port,
            verbose=True
        )

    except KeyboardInterrupt:
        print("-----")
        print("Exiting...")
        print("-----")

if __name__ == "__main__":
    main()