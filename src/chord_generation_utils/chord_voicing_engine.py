import numpy as np

class ChordVoicingEngine:
    def __init__(self, lower_octave_threshold: float = 0.25) -> None:
        """
        An action-focussed class which receives a Chord object and returns the voiced chord as a list[list[int, int]]
        """
        
        self.LOWER_OCTAVE_THESHOLD = lower_octave_threshold

        self.intensity_to_root_pitch = {
            0.66 : [36, 48],
            0.5  : [42, 54],
            0.33 : [48, 60],
            0    : [54, 66]
        }

    def get_voiced_chord(self, chord: any, intensity: float) -> list[list[int, int]]:
        """
        Voice a Chord object from its chroma histogram based on the intensity parameter.
        
        Params:
            chord: Chord - the chord object to be voiced
            intensity: float 0-1 - the intensity of the chord, used for mapping

        Returns:
            list[list[int, int]] - a list of MIDI notes as [MIDI pitch, MIDI velocity] pairs
        """
        
        voiced_chord = []

        chroma_histogram = chord.get_thresholded_chroma_histogram()

        # root_chroma = np.argmax(chroma_histogram)[0]

        for chroma_pitch, chroma_amount in enumerate(chroma_histogram):

            if chroma_amount >= chord.CHORD_NOTE_THRESHOLD: # Only do code block for valid chord notes above the threshold

                if intensity >= 0.45:
                    # Calculate actual pitch
                    if chroma_amount > self.LOWER_OCTAVE_THESHOLD:
                        midi_note_number = 36 + chroma_pitch + chord.TONIC
                    else:
                        midi_note_number = 60 + chroma_pitch + chord.TONIC
                
                elif intensity >= 0.25:
                    # Calculate actual pitch
                    if chroma_amount > self.LOWER_OCTAVE_THESHOLD:
                        midi_note_number = 48 + chroma_pitch + chord.TONIC
                    else:
                        midi_note_number = 60 + chroma_pitch + chord.TONIC

                else:
                    # Calculate actual pitch
                    if chroma_amount > self.LOWER_OCTAVE_THESHOLD:
                        midi_note_number = 48 + chroma_pitch + chord.TONIC
                    else:
                        midi_note_number = 72 + chroma_pitch + chord.TONIC
                
                # Calculate velocity
                velocity = 30 + int(intensity * 97)
                
                # Add to voiced_chord list of lists
                voiced_chord.append([midi_note_number, velocity])

        return voiced_chord