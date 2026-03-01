from music21 import stream, instrument, chord, analysis, midi, note, scale
from AstroMusicGenerator import AstroMusicGenerator, EnhancedAstroMusicGenerator
import numpy as np
import random
import copy
import multiprocessing
import logging



class AstroChoirGenerator:
    def __init__(self, fits_paths):
        self.fits_paths = fits_paths
        self.score = stream.Score()
        self._key_analysis_cache = {}

    def generate_choir(self, num_voices=4):
        """Generate multi-voice choir"""
        self.score = stream.Score()
        all_melodies = [self._generate_extended_melody(path) for path in self.fits_paths]
    
        voice_ranges = {
            'soprano': (72, 84),
            'alto': (60, 72),
            'tenor': (48, 60),
            'bass': (36, 48)
        }
    
        # Assign independent instruments to each voice
        voice_instruments = {
            'soprano': instrument.Soprano(),
            'alto': instrument.Alto(),
            'tenor': instrument.Tenor(),
            'bass': instrument.Bass()
        }
    
        for i, voice_name in enumerate(voice_ranges.keys()):
            melody = all_melodies[i % len(all_melodies)]
            transposed = self._adaptive_transpose(melody, list(voice_ranges.values())[i])
        
            # Create new part and add elements sequentially
            part = stream.Part(voice_instruments[voice_name])
            for element in transposed.flat.elements:
                if isinstance(element, (note.Note, note.Rest)):
                    part.append(element)
        
            self.score.insert(0, part)
    
        return self.score

    def _generate_extended_melody(self, path):
        """Generate and extend melody"""
        gen = EnhancedAstroMusicGenerator(path)
        melody = gen.generate_melody(gen.data, instrument.Violin())
        
        extended = self._markov_extend(melody, factor=2)
        varied = self._motif_variation(extended)
        return varied

    def _adaptive_transpose(self, melody, midi_range):
        """Smart transposition to fit vocal range"""
        """Add empty melody detection"""
        if not list(melody.flatten().notes):
            # Generate a default melody if empty
            default_scale = scale.MajorScale('C4')
            for _ in range(8):
                melody.append(note.Note(default_scale.pitches[np.random.randint(7)]))

        # Create deepcopy of melody to avoid modifying original
        melody_copy = copy.deepcopy(melody)
        # Perform operations on the copied object
        current_pitches = [n.pitch.midi for n in melody_copy.flatten().notes if isinstance(n, note.Note)]
        median_pitch = np.median(current_pitches) if current_pitches else 60
        
        # Calculate transpose value 
        target_center = (midi_range[0] + midi_range[1]) / 2
        transpose = int(target_center - median_pitch)
        if not melody:
            raise ValueError("Melody is empty, cannot transpose.")  
        return melody_copy.transpose(transpose)


    def _markov_extend(self, melody, factor=2):
        """Markov chain melody extension (safe version)"""
        import markovify
        notes = [str(n.pitch.midi) for n in melody.flatten().notes if isinstance(n, note.Note)]
        if len(notes) < 4:
            return melody
    
        corpus = " ".join(notes)
        try:
            text_model = markovify.Text(corpus, state_size=2)
        except:
            return melody  # Return a default note to prevent empty melody

        extended_notes = []
        max_attempts = 50
        attempt = 0
        while len(extended_notes) < len(notes) * factor and attempt < max_attempts:
            sentence = text_model.make_sentence()
            if len(extended_notes) < 20:  # Ensure at least 20 notes
                extended_notes = notes * 2  # Repeat original notes
            if sentence:
                extended_notes.extend(sentence.split())
            attempt += 1
    
        if not extended_notes:
            # Return a default note if empty
            n = note.Note(60)
            n.duration.quarterLength = 0.5
            return stream.Part([n])
            
        extended_stream = stream.Part()
        for midi_str in extended_notes[:len(notes)*factor]:
            try:
                n = note.Note(int(midi_str))
                n.duration.quarterLength = 0.5
                extended_stream.append(n)
            except:
                continue  # Skip invalid notes
        return extended_stream

    def _motif_variation(self, melody):
        """Motif variation algorithm"""
        new_part = stream.Part()
    
        # Extract first 4 notes as motif
        motif = [n for n in melody.flat.notes[:4] if isinstance(n, note.Note)]
        if not motif:
            return melody

        # Add original motif
        for note_obj in motif:
            new_part.append(note_obj)
    
        # Inversion variation
        inverted = [n.transpose(-5) for n in motif]
        for note_obj in inverted:
            new_part.append(note_obj)
    
        # Rhythm scaling (ensure only notes/rests are added)
        temp_stream = stream.Stream()
        for note_obj in motif:
            temp_stream.append(note_obj)
        scaled_stream = temp_stream.scaleDurations(1.5)
    
        # Tranverse variation and add to new part
        for element in scaled_stream.flat.elements:
            if isinstance(element, (note.Note, note.Rest)):
                new_part.append(element)
    
        # Process subsequent note variations
        for element in melody.notes[4:]:
            if isinstance(element, (note.Note, note.Rest)):
                if isinstance(element, note.Note) and random.random() < 0.3:
                    new_part.append(element.transpose(2))
                else:
                    new_part.append(element)
    
        return new_part
    
    def export_choir_midi(self, filename):
        """Export choir MIDI"""
        self.score.write('midi', filename)
        # Add reverb effect
        mf = midi.MidiFile()
        mf.open(filename)
        mf.add_reverb(level=0.6)  
        mf.save(filename)
        

# Upgrade multi-celestial processing
class EnhancedAstroChoirGenerator(AstroChoirGenerator):
    def generate_massive_choir(self, max_stars=997):
        # Dynamic chunking
        from itertools import islice
        
        """Large-scale stellar choir generation"""
        pool = multiprocessing.Pool(processes=8)
        try:
            chunks = [self.fits_paths[i:i+10] for i in range(0, len(self.fits_paths), 10)] 
            with multiprocessing.Pool(processes=4) as pool:  # Reduce number of processes
                melodies = pool.map(self._parallel_process_star, chunks)
        finally:
            pool.close()
        
        # Polyphonic score creation
        score = self._create_polyphonic_score(melodies)
        return score
    
    def _select_instrument_by_line(self, data):
        """Select instrument based on spectral lines"""
        if 'emission_lines' in data and any('Hα' in line for line in data['emission_lines']):
            return instrument.Violin()
        return instrument.Piano()
    
    def _create_polyphonic_score(self, melodies):
        """Polyphonic score creation (with automatic instrument assignment)"""
        score = stream.Score()
        voice_ranges = [(36,48), (48,60), (60,72), (72,84)]
        
        for i, melody in enumerate(melodies):
            # Obtain data from generator
            part_data = melody.generator.data if hasattr(melody, 'generator') else {}
            part = stream.Part(self._select_instrument_by_line(part_data))
            # Assign vocal range automatically
            transposed = self._adaptive_transpose(melody, voice_ranges[i%4])
            # Add spatialization effect
            pan = (i/len(melodies))*2 - 1  # Distribute panning from left to right
            transposed.getInstrument().pan = pan
            part.append(transposed)
            score.insert(0, part)
        
        return score

    def _parallel_process_star(self, path):
        try:
            """Parallel processing of single star"""
            gen = EnhancedAstroMusicGenerator(path)
            return gen.generate_full_composition()
        except Exception as e:
            logging.error(f"Process {path} Fail: {str(e)}")
            return stream.Score()  # Return empty score to prevent crash
    
