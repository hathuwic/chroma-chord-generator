import pretty_midi as pm
import numpy as np
import pandas as pd
import time

class MIDIFileProcessor:
    
    def __init__(self, key_classifier: any) -> None:
        self.MELODY_KEYWORDS = [
            "solo",
            "melody",
            "vocal",
            "vox",
            "voice",
            "soprano",
            "lead",
            "1st",
            "first"]
        
        self.MELODY_EXCLUDED = [
            "drum",
            "kick",
            "snare",
            "tom",
            "hat",
            "hh",
            "cymbal",
            "ride",
            "crash"
        ]

        self.key_classifier = key_classifier

        self.HARMONY_NOTE_CHORD_THRESHOLD = 1/16

# MELODY INSTRUMENT

    def __get_instrument_average_pitch(self, instrument: pm.Instrument) -> float:
        start = time.time()
        notes_count = 0
        note_numbers_total = 0
        for note in instrument.notes:
            notes_count += 1
            note_numbers_total += note.pitch
        # In case of no notes, add 1 to avoid div by 0 error
        if notes_count == 0:
            notes_count = 1
        avg_pitch = note_numbers_total / notes_count
        # print(f"  Computed average pitch in {time.time()-start} secs")
        return avg_pitch

    def __get_instrument_overlap(self, instrument: pm.Instrument) -> float:
        """Get the percentage overlap (0-1) of the given instrument."""
        start = time.time()
        
        this_instrument_note_overlap_times = np.zeros((1))
        notes_dur_sum = 0

        # Check each note against all other notes and calculate overlap percentage (0-1)
        for current_note_i, current_note in enumerate(instrument.notes): 
            current_note_dur = current_note.get_duration()
            notes_dur_sum += current_note_dur # Add note duration to sum of durations

            # List containing (2,) overlap arrays for each overlapping note
            current_note_overlaps = []

            # Loop through all notes to if overlapping
            for check_note_i, check_note in enumerate(instrument.notes):

                # Initialise note_overlap_region array to contain [overlap_start, overlap_end]
                note_overlap_region = np.zeros((2))

                # Bypass if comparing current note against itself
                if current_note_i == check_note_i:
                    continue
                # Bypass all non-overlapping notes
                elif current_note.start >= check_note.end or current_note.end <= check_note.start: 
                    pass  
                # If note to check overlaps whole of current note, continue with 1 i.e., entire note is overlapped
                elif current_note.start >=check_note.start and current_note.end <= check_note.end:
                    note_overlap_region[1] = current_note_dur
                # If partial overlap
                else:
                    # If note is overlapped at the start i.e., use note start and note_to_check end
                    if current_note.start >= check_note.start and current_note.end > check_note.end:
                        # print("    Start overlapped.")
                        note_overlap_region[0] = 0 # Overlapping from start of note
                        note_overlap_region[1] = current_note.end - check_note.end # Overlapping to this many seconds through the note
                    # If note is overlapped at the end i.e., use note_to_check start and note end
                    elif current_note.start < check_note.start and current_note.end <= check_note.end:
                        # print("    End overlapped.")
                        note_overlap_region[0] = check_note.start - current_note.start # Overlapping from this many seconds through note
                        note_overlap_region[1] = current_note_dur # Overlapping to end of note
                    # If note_to_check is wholly contained within the current note    
                    elif current_note.start < check_note.start and current_note.end > check_note.end:
                        # print("    Overlap contained inside note.")
                        note_overlap_region[0] = check_note.start - current_note.start # Overlapping from this many seconds through note
                        note_overlap_region[1] = current_note.end - check_note.end # Overlapping to this many seconds through note
                    # Error case
                    else:
                        print("  ERROR: No partial overlap conditions met.")

                # Append overlap percentage for current note to note_overlaps array
                current_note_overlaps.append(note_overlap_region)

            # Initialise overlapped time for this note to 0.
            current_note_overlap_time = 0

            # Loop through overlaps to find most overlap of current note
            for overlap in current_note_overlaps:
                # If note is entirely overlapped, overlap is note duration and then exit loop as max overlap duration found
                if overlap[0] == 0 and overlap[1] == current_note_dur:
                    current_note_overlap_time = current_note_dur
                    break
                # If no overlap, move onto next note
                elif overlap[0] == 0 and overlap[1] == 0:
                    continue
                # If partial overlap, check if bigger than current overlap and update if so
                elif (overlap[1] - overlap[0]) < current_note_dur:
                    current_note_overlap_time = overlap[1] - overlap[0]   
                # Error case
                else:
                    print("  ERROR: no cases met for overlap.")

            # Append current note overlap time to array of overlap times for this instrument
            this_instrument_note_overlap_times = np.append(this_instrument_note_overlap_times, current_note_overlap_time)
            if current_note_overlap_time > current_note_dur: # Error case
                print("  ERROR: overlap duration longer than note duration.")
            
            # Calculate overlap percentage in range 0-1
            this_instrument_overlap_percentage = this_instrument_note_overlap_times.sum() / notes_dur_sum

            # print(f"  Computed overlap in {time.time()-start} secs")
            
            return this_instrument_overlap_percentage

    def __get_instrument_stats(self, instrument: pm.Instrument) -> tuple[float, float]:
        avg_pitch = self.__get_instrument_average_pitch(instrument)
        overlap = self.__get_instrument_overlap(instrument)
        return (avg_pitch, overlap)

    def __choose_melody_instrument_from_options(self, instruments: dict) -> tuple[pm.Instrument, float, float]:
        highest_avg_pitch = 0
        for instrument, stats in instruments.items():
            if stats[0] > highest_avg_pitch:
                highest_avg_pitch = stats[0]
                melody_instrument = instrument
        return (
            melody_instrument,
            instruments[melody_instrument][0],
            instruments[melody_instrument][1]
        )

    def __is_valid_melody_instrument(self, instrument: pm.Instrument) -> bool:   
        """
        Check whether an instrument is a valid option for a melody instrument.
        Returns False for drums and might-be drums, otherwise returns True."""
        # Check for drum track
        if instrument.is_drum:
            return False
        # Check for other unlikely candidates i.e., things named after drums
        for excluded_word in self.MELODY_EXCLUDED:
            if excluded_word in instrument.name.lower():
                return False
        return True

    def get_melody_instrument(self, midi_file: pm.PrettyMIDI) -> pm.Instrument:
        print(f"    # of tracks in MIDI file: {len(midi_file.instruments)}")

        melody_instrument_option_stats = {}
        melody_instrument_options = []
        named_melody_instruments = []
        
        # Get a list of possible melody instruments, and a list of named instruments
        for instrument in midi_file.instruments:
            # Only continue processing valid melody instrument options
            if self.__is_valid_melody_instrument(instrument):
                melody_instrument_options.append(instrument)
                # Check if named with melody keywords, add to special list if so
                for keyword in self.MELODY_KEYWORDS:
                    if keyword in instrument.name.lower():
                        named_melody_instruments.append(instrument)

        # If named melody instrument(s) found, handle the list
        if len(named_melody_instruments) >= 1:
            # If only one option named option, assume that as melody instrument and return it
            if len(named_melody_instruments) == 1:
                melody_instrument = named_melody_instruments[0]
            # If multiple named options
            elif len(named_melody_instruments) > 1:
                for instrument in named_melody_instruments:
                    avg_pitch, overlap = self.__get_instrument_stats(instrument)
                    melody_instrument_option_stats[instrument] = (avg_pitch, overlap)
                melody_instrument, avg_pitch, overlap = self.__choose_melody_instrument_from_options(melody_instrument_option_stats)
        
        # If no named melody instruments found, check all valid from melody_instrument_options
        else:
            # Get avg pitch and overlap amount for each possible melody instrument
            for instrument in melody_instrument_options:
                avg_pitch, overlap = self.__get_instrument_stats(instrument)
                melody_instrument_option_stats[instrument] = (avg_pitch, overlap)

            # Choose melody instrument from options based on stats
            melody_instrument, avg_pitch, overlap = self.__choose_melody_instrument_from_options(melody_instrument_option_stats)

        print(f"    Found melody instrument: {melody_instrument.name}")
        
        return melody_instrument

# ADDING TO DATAFRAME

    def __get_harmony_instruments_list(self, midi_file: pm.PrettyMIDI, melody_instrument: pm.Instrument) -> list[pm.Instrument]:
        # Get list of harmony instruments by removing melody instrument and all drum tracks
        harmony_instruments = midi_file.instruments.copy()
        for instrument in harmony_instruments:
            if instrument.is_drum or instrument is melody_instrument:
                harmony_instruments.remove(instrument)

        return harmony_instruments

    def __get_ks_downbeats_list(self, midi_file: pm.PrettyMIDI, ks_start: float, ks_end: float) -> list[float]:
        downbeats = midi_file.get_downbeats(start_time=ks_start)
        for db_idx, downbeat in enumerate(downbeats):
            if downbeat > ks_end:
                downbeats = downbeats[:db_idx+1]
                break

        return downbeats

    def __get_longest_note_in_bar(self,
                                      melody_instrument: pm.Instrument,
                                      bar_start: float,
                                      bar_end: float,
                                      bar_duration: float) -> any:
        """Find the longest note within the given bar of the given instrument."""
        
        # Init variables for the maximum bar coverage of a single note & selected note
        melody_note_max_coverage = 0
        selected_melody_note = None

        # Loop through melody notes in bar
        for melody_note in melody_instrument.notes:
            # Check if note in bar
            if bar_start <= melody_note.start <= bar_end or bar_start <= melody_note.end <= bar_end:
                melody_note_coverage = melody_note.get_duration() / bar_duration
                if melody_note_coverage > melody_note_max_coverage:
                    melody_note_max_coverage = melody_note_coverage
                    selected_melody_note = melody_note

        return selected_melody_note

    def __is_note_in_bar(self, note: pm.Note, bar_start: float, bar_end: float) -> bool:
        if bar_start <= note.start <= bar_end or bar_start <= note.end <= bar_end:
            return True
        else:
            return False

    def __get_bar_chroma_histogram(self,
                                   harmony_instruments: list[pm.Instrument],
                                   bar_start: float,
                                   bar_end:float,
                                   bar_duration: float,
                                   tonic: int) -> np.ndarray:
        
        """Compute the chroma histogram for the given list of instruments
        within the given bar start and end times."""

        # Init array to hold chroma histogram, and variable to hold total sum of bar
        bar_harmony_chroma = np.zeros(12)
        bar_coverage_sum = 0

        # Loop over harmony instruments
        for harmony_instrument in harmony_instruments:
            
            # Loop over note within instrument
            for harmony_note in harmony_instrument.notes:
                
                # Check if note is in the current bar
                if self.__is_note_in_bar(harmony_note, bar_start, bar_end):
                    harmony_note_coverage = harmony_note.get_duration() / bar_duration
                    
                    # If coverage is sufficiently high, then include in the chroma histogram
                    if harmony_note_coverage >= self.HARMONY_NOTE_CHORD_THRESHOLD:
                        harmony_note_chroma = (harmony_note.pitch - tonic) % 12
                        bar_harmony_chroma[harmony_note_chroma] += harmony_note_coverage
                        bar_coverage_sum += harmony_note_coverage
        
        # If coverage by harmony notes is greater > 0, i.e., there are harmony notes in the bar,
        # normalise to sum to 1
        if bar_coverage_sum > 0:
            bar_harmony_chroma = bar_harmony_chroma / bar_coverage_sum
        
        return bar_harmony_chroma

    def __clean_midi_file_chords_array(self, midi_file_chords_array: np.ndarray) -> np.ndarray:
        # Init list holding indices of chords to delete from array
        chords_to_delete = []

        # Find indices of chords to delete
        for idx, melody_chord_pair in enumerate(midi_file_chords_array):
            if melody_chord_pair[1:].sum() == 0:
                chords_to_delete.append(idx)

        # Delete chords using indices
        if len(chords_to_delete) != 0:
            midi_file_chords_array = np.delete(midi_file_chords_array, chords_to_delete, axis=0)
            print(f"    Purging empty chords...")
            print(f"    Chords after removal: {midi_file_chords_array.shape[0]} (-{len(chords_to_delete)})")

        return midi_file_chords_array

    def get_chords_as_array(self,
                            midi_file: pm.PrettyMIDI,
                            melody_instrument: pm.Instrument,
                            key_signatures: list[pm.KeySignature]) -> None:
        """Generate an (x,13) Numpy array representing all the chords in the provided MIDI file."""

        # Get list of harmony instruments by removing melody instrument and all drum tracks
        harmony_instruments = self.__get_harmony_instruments_list(midi_file, melody_instrument)

        # Initialise array for holding all chords and melody chroma for current MIDI file
        midi_file_chords_array = np.zeros((1, 13))

        for ks_idx, ks in enumerate(key_signatures):
            
            # Get tonic of key / relative major of key
            if ks.key_number > 11:
                tonic = (ks.key_number + 3) % 12
                #continue # <- THIS EXCLUDES MINOR KEY SIGNATURES WHEN UNCOMMENTED
            else:
                tonic = ks.key_number

            # Get time range for current key signature (just end of file if only one ks or last ks in list)
            ks_start = ks.time
            if len(key_signatures) <= ks_idx+1:
                ks_end = midi_file.get_end_time()
            else:
                ks_end = key_signatures[ks_idx+1].time

            # Get list of downbeats in the current key signature range
            downbeats = self.__get_ks_downbeats_list(midi_file, ks_start, ks_end)

            # Init array to hold chroma histogram values, will be appended to df once per KS
            ks_chroma_histograms = np.full((downbeats.size, 13), float(0))

            # Loop through bars
            for bar_idx, bar_start in enumerate(downbeats):
                
                # Get end of bar i.e., end of range to look at
                # If last bar, set end time as the length of the previous interval after the last downbeat
                if bar_idx < downbeats.size - 1:
                    bar_end = downbeats[bar_idx+1]
                else:
                    bar_end = bar_start + (bar_start - downbeats[bar_idx-1])
                
                # Get bar duration in seconds
                bar_duration = bar_end - bar_start

                # Find longest melody note in current bar
                selected_melody_note = self.__get_longest_note_in_bar(melody_instrument, bar_start, bar_end, bar_duration)

                # Continue with found melody note
                if selected_melody_note != None:
                    # Find chroma of longest note, add to array in 1st index of corresponding row
                    melody_chroma = (selected_melody_note.pitch - tonic) % 12
                    ks_chroma_histograms[bar_idx, 0] = melody_chroma

                    # Compute harmony chroma histogram for current bar
                    bar_harmony_chroma = self.__get_bar_chroma_histogram(
                        harmony_instruments,
                        bar_start,
                        bar_end,
                        bar_duration,
                        tonic)

                    # Update ks chords array with current bar histogram
                    for chroma_idx, chroma in enumerate(bar_harmony_chroma):
                        ks_chroma_histograms[bar_idx, chroma_idx+1] = chroma
            
            midi_file_chords_array = np.append(midi_file_chords_array, ks_chroma_histograms, axis=0)

        # Trim first empty row
        midi_file_chords_array = midi_file_chords_array[1:, :] 
        
        print(f"    Generated {midi_file_chords_array.shape[0]} chroma histograms from file.")
        
        # Remove any rows with a chroma histogram summing to 0, i.e., no harmony notes in the bar
        midi_file_chords_array = self.__clean_midi_file_chords_array(midi_file_chords_array)
        
        # Return chord array with empty chords removed
        return midi_file_chords_array

# KEY SIGNATURE EXTRACTION / PREDICTION

    def print_key_signatures_in_file(self, midi_file: pm.PrettyMIDI) -> None:
        if len(midi_file.key_signature_changes) > 0:
            for i, ks in enumerate(midi_file.key_signature_changes):
                print(f"    Key sig {i+1}: {ks}")
        else:
            print("    NO KEY SIGNATURES")

    def __predict_key_signature(self, midi_file: pm.PrettyMIDI) -> list[pm.KeySignature]:
        midi_file_chroma_histogram = midi_file.get_pitch_class_histogram(normalize=True, use_duration=True).reshape(1,-1)
        key_signature = int(self.key_classifier.predict(midi_file_chroma_histogram)[0]) # Prediction returns 1x1 array, so take first element and make int
        return pm.KeySignature(key_signature, 0)

    def __is_valid_key_signature(self, ks: pm.KeySignature) -> bool:
        if ks.key_number == 0 and ks.time == 0:
            return False
        return True

    def __clean_key_signature_list(self, key_signatures: list[pm.KeySignature]) -> list[pm.KeySignature]:
        key_signatures_cleaned = []
        previous_ks = -1
        for ks in key_signatures:
            if ks.key_number != previous_ks:
                key_signatures_cleaned.append(ks)
            previous_ks = ks.key_number
        
        return key_signatures_cleaned

    def get_key_signatures(self, midi_file: pm.PrettyMIDI) -> list[pm.KeySignature]:
        """
        Get a list of key signatures for the MIDI file, either by getting
        directly from the file or predicting it if no key signature in included.
        """
        
        # If none, predict the key from the chroma histogram
        if len(midi_file.key_signature_changes) == 0:
            return [self.__predict_key_signature(midi_file)]
        
        # If one key signature
        elif len(midi_file.key_signature_changes) == 1:
            # If valid ks, return as list
            if self.__is_valid_key_signature(midi_file.key_signature_changes[0]):
                return midi_file.key_signature_changes
            # If not valid, predict and return
            else:
                return [self.__predict_key_signature(midi_file)]
        
        # If multiple key signatures in file, clean up if necessary then return
        else:
            return self.__clean_key_signature_list(midi_file.key_signature_changes)