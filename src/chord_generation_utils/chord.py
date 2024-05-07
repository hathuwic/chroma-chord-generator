import numpy as np
from .chord_voicing_engine import ChordVoicingEngine

class Chord:
    def __init__(self,
                 chroma_histogram: np.ndarray,
                 tonic: int = 0,
                 chord_note_threshold: float = 0.1
        ) -> None:
        
        """
        A data-focussed object which holds everything to do
        with a chroma histogram and its associated voiced chord.

        Parameters:
            chroma_histogram:       np.array of size 12. The unnormalised chroma histogram for this chord
            tonic:                  int (0-11) (default 0). The tonic associated with this chord, used when voicing the chord.
            chord_note_threshold:   float (0-1) (default 0.1). The amount above which a chroma pitch must be present in the histogram to be present in the voiced chord, or thresholded histogram.

        Returns:
            An instance of the chord object.
        """

        # Handle received tonic
        if type(tonic) == int:
            if 0 <= tonic <= 11:
                self.TONIC = tonic
            else:
                raise ValueError(f"Tonic must be int in range 0-11. Received {tonic}.")
        else:
            raise ValueError(f"Tonic must be of type int in range 0-11. Received {tonic} (type: {type(tonic)})")
        
        # Handle received chroma histogram
        if chroma_histogram.size == 12:
            self.chroma_histogram = chroma_histogram
        else:
            raise ValueError(f"Chroma histogram must be a Numpy array of shape (12,). Received {chroma_histogram} (type: {type(chroma_histogram)})")
        
        # Handle recieved chord note threshold value
        if type(chord_note_threshold) == float and 0 < chord_note_threshold <= 1:
            self.CHORD_NOTE_THRESHOLD = chord_note_threshold
        else:
            raise ValueError(f"Chord note threshold must be a float in range 0-1. Received {chord_note_threshold} (type: {type(chord_note_threshold)})")
        
        self.chord_voicing_engine = ChordVoicingEngine()

    def get_voiced_chord(self, intensity: float = 0) -> list[list[int, int]]:
        """Voices the chord using self.chord_voicing_engine and returns it as an a list of [pitch, velocity] pairs."""

        return self.chord_voicing_engine.get_voiced_chord(self, intensity=intensity)
    
    def get_thresholded_chroma_histogram(self) -> np.ndarray[float]:
        """Returns the thresholded version of the chroma histogram based on self.CHORD_NOTE_THRESHOLD."""
        
        # Init array to hold thresholded chroma histogram
        thresholded_chroma_histogram = np.zeros(12)
        
        # Iterate over chroma histogram to find notes that are above the threshold
        # When found, add values to thresholded chroma histogram array
        for i, chroma_amount in enumerate(self.chroma_histogram):
            if chroma_amount > self.CHORD_NOTE_THRESHOLD:
                thresholded_chroma_histogram[i] = chroma_amount
        
        # Normalise thresholded array to sum to 1
        if thresholded_chroma_histogram.sum() != 0:
            thresholded_chroma_histogram = thresholded_chroma_histogram / thresholded_chroma_histogram.sum()

        return thresholded_chroma_histogram

    def get_unthresholded_chroma_histogram(self) -> np.ndarray[float]:
        """Returns the un-thresholded version of the chroma histogram, i.e., that provided when creating the instance."""
        return self.chroma_histogram

if __name__ == "__main__":

    # Simple tests for object

    chroma_histogram = np.asarray([0.3, 0, 0, 0.05, 0.15, 0.05, 0, 0.4, 0, 0, 0, 0.1,])

    chord = Chord(chroma_histogram, tonic=3, chord_note_threshold=0.14)

    print(chord.get_unthresholded_chroma_histogram())
    print(chord.get_thresholded_chroma_histogram())
    print(chord.get_voiced_chord())