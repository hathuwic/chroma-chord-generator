import numpy as np

class InputSequence:
    def __init__(
            self,
            sequence_length: int = 8,
            init_type: str = "rand",
            update_direction: str = "append"
        ) -> None:

        """
        A data-focussed class which keeps track of the input sequence for the chord prediction model.
        It is used internally as a component in the ChordGenerator object.
        """

        # Store the sequence length to be used
        self.SEQUENCE_LENGTH = sequence_length

        # Init melody_chroma_history as list of zeros
        self.melody_chroma_history = np.zeros(self.SEQUENCE_LENGTH)

        # Init the chroma histogram history using random numbers
        if init_type == "rand":
            self.chroma_histogram_history = np.random.rand(self.SEQUENCE_LENGTH, 12) 
            for i, chord in enumerate(self.chroma_histogram_history): # Iterate over chroma histograms
                self.chroma_histogram_history[i] = chord/chord.sum()
        else:
            raise NotImplementedError("Only possible init_type is 'rand'")
        
        if update_direction == "append" or update_direction == "prepend":
            self.UPDATE_DIRECTION = update_direction
        else:
            raise ValueError(f"Update direction must be 'append' or 'prepend'.")

    def update_melody_chroma_history(self, melody_chroma: int) -> None:
        """Update the stored history of melody chroma values with a new value, dropping the oldest one."""
        if self.UPDATE_DIRECTION == "append":
            self.melody_chroma_history = np.append(self.melody_chroma_history, melody_chroma)
            self.melody_chroma_history = self.melody_chroma_history[1:]
        elif self.UPDATE_DIRECTION == "prepend":
            self.melody_chroma_history = np.append(melody_chroma, self.melody_chroma_history)
            self.melody_chroma_history = self.melody_chroma_history[:-1]

    def update_chroma_histogram_history(self, chroma_histogram: np.ndarray[float]) -> None:
        """Update the stored history of chroma histograms with a new histogram, dropping the oldest one."""
        if self.UPDATE_DIRECTION == "append":
            self.chroma_histogram_history = np.append(self.chroma_histogram_history, chroma_histogram.reshape(1,12), axis=0) # Add new value to array
            self.chroma_histogram_history = self.chroma_histogram_history[1:,:] # Drop the first i.e., oldest value in the array
        elif self.UPDATE_DIRECTION == "prepend":
            self.chroma_histogram_history = np.append(chroma_histogram.reshape(1,12), self.chroma_histogram_history, axis=0) # Add new value to array
            self.chroma_histogram_history = self.chroma_histogram_history[:-1,:] # Drop the last i.e., oldest value in the array

    def get(self) -> np.ndarray[float]:
        """Return the currently stored input sequence as a Numpy array of shape (1, self.SEQUENCE_LENGTH, 13)."""
        
        # Reshape melody_chroma_history to have same dimensions as chroma_histogram_history
        melody_chroma_history_reshaped = self.melody_chroma_history.reshape(self.SEQUENCE_LENGTH, 1)
        
        # Append melody chroma history to chroma histogram history to create array of correct shape for Tensorflow model
        input_sequence = np.append(melody_chroma_history_reshaped, self.chroma_histogram_history, axis=1).reshape(1, self.SEQUENCE_LENGTH, 13)
        
        return input_sequence
    
if __name__ == "__main__":

    # Tests of the object

    input_sequence = InputSequence()

    print(input_sequence.get())

    input_sequence.update_melody_chroma_history(3)

    print(input_sequence.get())

    chroma_histogram = np.asarray([0.3, 0, 0, 0.05, 0.15, 0.05, 0, 0.4, 0, 0, 0, 0.1,])
    input_sequence.update_chroma_histogram_history(chroma_histogram)

    print(input_sequence.get())