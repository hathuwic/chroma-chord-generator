from pythonosc import udp_client, osc_server
from pythonosc.dispatcher import Dispatcher
from .chord_generator import ChordGenerator
from .chord import Chord
import time

class OSCHandler:
    def __init__(self, chord_generator: ChordGenerator, ip: str = "127.0.0.1", server_port: int = 10000, client_port: int = 11000, verbose: bool = False) -> None:
        
        # Create necessary attributes
        self.chord_generator = chord_generator
        self.IP = ip
        self.CLIENT_PORT = client_port
        self.SERVER_PORT = server_port
        self.VERBOSE = verbose
        
        # Init previous chord variable, used for cancelling active notes before starting a new chord.
        self.previous_chord = None

        # Start OSC client
        self.client = udp_client.SimpleUDPClient("127.0.0.1", self.CLIENT_PORT)
        print(f"Started OSC client at: {self.IP}:{self.CLIENT_PORT}")
        print("MIDI notes will be sent from this port as: list[list[pitch, velocity]].")
        print("-----")

        # Create dispatcher and map callbacks
        self.dispatcher = Dispatcher()
        self.dispatcher.map("/melody_note",                     self.handle_melody_note)
        self.dispatcher.map("/set_tonic",                       self.set_chord_generator_tonic_from_OSC)
        self.dispatcher.map("/set_threshold",                   self.set_chord_generator_chord_note_threshold_from_OSC)
        self.dispatcher.map("/set_threshold_input_sequence",    self.set_threshold_input_sequence_from_OSC)

        # Start OSC server (blocks)
        self.server = osc_server.BlockingOSCUDPServer((ip, server_port), self.dispatcher)
        print(f"Started OSC server at: {self.IP}:{self.SERVER_PORT}")
        print(f"Expecting list[MIDI pitch (0-127), intensity (0-1)] on this port.")
        print("-----")
        print("Awaiting MIDI pitches...")
        self.server.serve_forever() # Blocks forever

    def set_chord_generator_tonic_from_OSC(self, address: str, *args) -> None:
        if len(args) == 1:
            if type(args[0]) == int:
                self.chord_generator.set_tonic(args[0])
            else:
                raise TypeError(f"New tonic must be set using int type. Received {type(args[0])}")
        else:
            raise ValueError(f"list[int] required to set new tonic. Received list of length {len(args)}")
        
    def set_chord_generator_chord_note_threshold_from_OSC(self, address: str, *args) -> None:
        if len(args) == 1:
            if type(args[0]) == float or type(args[0]) == int:
                self.chord_generator.set_chord_note_threshold(args[0])
            else:
                raise TypeError(f"New chord note threshold must be set using float or int. Received {type(args[0])}")
        else:
            raise ValueError(f"list[int] or list[float] required to set new chord note threshold. Received list of length {len(args)}")
        
    def set_threshold_input_sequence_from_OSC(self, address: str, *args) -> None:
        if len(args) == 1:
            if type(args[0]) == int:
                if args[0] == 0:
                    new_threshold_state = False
                elif args[0] == 1:
                    new_threshold_state = True
                else:
                    raise ValueError(f"New threshold output state must be 0 or 1. Received {args[0]}")
                self.chord_generator.set_threshold_input_sequence(new_threshold_state)
            else:
                raise TypeError(f"New threshold output state must be int 0 or 1. Received {type(args[0])}")
        else:
            raise ValueError(f"New threshold state must be OSC message of list with length 1. Received list with length {len(args)}")

    def stop_chord(self, previous_chord: list[int, int]) -> None:
        if type(previous_chord) == list:
            for note in previous_chord:
                self.client.send_message("/note_off", [note[0], 0])
        else:
            if self.VERBOSE:
                print(f"Tried to cancel chord but found no valid notes. Skipping...")

    def handle_melody_note(self, address: str, *args) -> None:
        """
        Received a melody note as MIDI note number,
        uses the ChordGenerator object to predict a chord and
        then returns the notes of that chord back over OSC.
        """

        handler_start_time = time.time()

        # Stop notes from the previous chord
        self.stop_chord(self.previous_chord)

        # Predict the new chord
        melody_midi_note_number = args[0]
        chord = self.chord_generator.get_chord(melody_midi_note_number)

        # Send notes for new chord
        voiced_chord = chord.get_voiced_chord(intensity=args[1])
        for note in voiced_chord:
            self.client.send_message("/chord_note", note)
        
        # Send chroma histogram of predicted chord
        self.client.send_message("/histogram", chord.get_thresholded_chroma_histogram().tolist())
        
        # Print contents of chord if in verbose mode
        if self.VERBOSE:
            print(f"Took {round((time.time() - handler_start_time)*1000, 3)} ms to voice chord with {len(voiced_chord)} notes: {voiced_chord}")

        # Update previous chord with newly generated chord, for stopping notes on next callback of this method
        self.previous_chord = voiced_chord

if __name__ == "__main__":
    pass