import numpy as np
import tensorflow as tf
import logging
from .input_sequence import InputSequence
from .chord import Chord
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Gets TensorFlow to shut up...

class ChordGenerator:
    def __init__(
            self,
            model_path: str,
            sequence_length: int,
            tonic: int,
            chord_note_threshold: float = 0.1,
            threshold_input_sequence: bool = True,
            update_direction: str = "append",
            log_level = logging.INFO
        ) -> None:

        # Handle creation parameters
        self.MODEL_PATH = model_path
        self.INPUT_SEQUENCE_LENGTH = sequence_length
        self.TONIC = tonic
        self.CHORD_NOTE_THRESHOLD = chord_note_threshold
        self.THRESHOLD_INPUT_SEQUENCE = threshold_input_sequence

        # Create internal logger
        self.logger = logging.Logger("chord_generator", log_level)

        # Setup methods
        self.load_model(self.MODEL_PATH)
        
        # Initialise the InputSequence object
        self.input_sequence = InputSequence(sequence_length=self.INPUT_SEQUENCE_LENGTH,
                                            init_type = "rand", update_direction=update_direction)

    def load_model(self, model_path) -> None:
        model_load_start_time = time.time()
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.logger.info(f"Took {round(time.time() - model_load_start_time, 3)} secs to successfully loaded model from: {model_path}")

        except Exception as e:
            self.logger.error(f"Unable to load model from: {model_path}")
            raise e

    def get_chord(self, melody_note_midi_number: int) -> Chord:
        
        chord_prediction_start_time = time.time()

        # Get melody chroma of received note
        melody_chroma = (melody_note_midi_number - self.TONIC) % 12

        # Update input sequence with new melody chroma
        self.input_sequence.update_melody_chroma_history(melody_chroma)

        # Predict chroma histogram using updated input_sequence
        chroma_histogram = self.model(self.input_sequence.get()).numpy()[0]

        # Normalise to ensure histogram sums to as expected
        chroma_histogram = chroma_histogram / chroma_histogram.sum()
        
        #Create Chord object to hold the new chroma histogram
        chord = Chord(chroma_histogram,
                      tonic=self.TONIC,
                      chord_note_threshold=self.CHORD_NOTE_THRESHOLD)
        
        # Update input sequence with the new chroma histogram, either thresholded or not
        if self.THRESHOLD_INPUT_SEQUENCE:
            self.input_sequence.update_chroma_histogram_history(chord.get_thresholded_chroma_histogram())
        else:
            self.input_sequence.update_chroma_histogram_history(chord.get_unthresholded_chroma_histogram())
        
        self.logger.info(f"Generated new chord in {round((time.time() - chord_prediction_start_time)/1000, 3)} ms")

        return chord
    
    def set_tonic(self, new_tonic: int) -> None:
        self.TONIC = new_tonic

    def set_chord_note_threshold(self, new_threshold: float) -> None:
        self.CHORD_NOTE_THRESHOLD = new_threshold
    
    def set_threshold_input_sequence(self, new_state: bool) -> None:
        self.THRESHOLD_INPUT_SEQUENCE = new_state

if __name__ == "__main__":

    chord_generator = ChordGenerator("C:/Users/jacke/OneDrive - Universitetet i Oslo/Thesis/thesis_repo/chord_generation/models/2024-02-22_15-32-12_chord_generator_sq-16_btch-64_lr-0.0001_dropout-0.25_epoch-12/",
                                     sequence_length=16,
                                     tonic=3,
                                     chord_note_threshold=0.1,
                                     threshold_input_sequence=True)
    
    chord = chord_generator.get_chord(77)
    print(chord.get_voiced_chord())