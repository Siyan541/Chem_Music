import numpy as np
from astropy.io import fits
from music21 import stream, note, instrument, scale, harmony, pitch, meter, tempo, dynamics
from music21 import expressions, duration, midi
from music21 import *
from music21.dynamics import Crescendo
import pywt
import logging
logging.basicConfig(level=logging.INFO)
from astropy.table import Table
from music21.instrument import SnareDrum, BassDrum
import os
from music21 import environment
from music21 import midi


# Configure core parameters
env = environment.Environment()
env['autoDownload'] = 'allow'  # Allow automatic downloading of necessary components

# Specify FluidSynth path
midi.realtime.fluidsynthPath = '/opt/homebrew/bin/fluidsynth'  # which fluidsynth to get actual path

# Explicitly specify SoundFont path when needed
SOUNDFONT_PATH = '/soundfonts/FluidR3_GM.sf2'

# --------------------------
# Astronomical parameter mapping configuration
# --------------------------
PARAM_MAPPING = {
    'wavelength': {
        'element': 'chord_complexity',
        'mapping': lambda wl: (np.var(wl)/1e6, 0, 3)  # Wavelength variance → chord tension
    },
    'flux_derivative': {
        'element': 'rhythm_density',
        'mapping': lambda df: (np.mean(np.abs(df))*100, 0, 1) # Flux derivative → rhythm density
    },
    'feh': {
        'element': 'scale_mode',
        'mapping': lambda feh: ('minor' if feh < -1 else 'lydian') # Metallicity [Fe/H] → scale mode
    },
    'z': {
        'element': ['transpose', 'tempo'],
        'mapping': lambda z: (int(z*12), 1.0 + z*0.5) # Redshift → transpose & tempo
    },
    'emission_lines': {
        'element': 'timbre_change',
        'mapping': lambda lines: {line: intensity for line, intensity in lines} # Emission lines → timbre changes
    }
}

# --------------------------
# Core music generation engine
# --------------------------
class AstroMusicGenerator:

    def __init__(self, fits_path):
        self.data = self._load_and_process_data(fits_path)
        self.score = stream.Score()
        self._apply_global_settings()
        self.key = key.Key('C') 
        self.spatial_pan = 0.0
        self.reverb_strength = 1.0

    def data_cleaning(self, flux):
            """Data cleaning function"""
            # Remove NaN or 0 flux values
            valid_mask = (np.isfinite(flux)) & (flux > 0)
            clean_flux = flux[valid_mask]

            # Handle all-invalid data case
            if len(clean_flux) == 0:
                clean_flux = np.zeros_like(flux)  # Fill with zeros

            # Fill remaining NaN
            clean_flux = np.nan_to_num(clean_flux, nan=np.nanmedian(clean_flux))

            # Flux normalization (prevent excessive volume)
            normalized_flux = (clean_flux - np.min(clean_flux)) / \
                     (np.max(clean_flux) - np.min(clean_flux) + 1e-8)  # pervent division by zero
            
            return normalized_flux

    def wavelet_denoising(self, flux, level=5):
        """
        Improved wavelet denoising ensuring consistent input/output length
        :param flux: original flux array
        :param level: wavelet denoising level
        :return: flux after denoising (same length as input)
        """
        # Original length
        original_len = len(flux)
        
        # Calculate padding length (make length multiple of 2^level)
        desired_len = int(np.ceil(original_len / (2**level)) * (2**level))
        pad_width = desired_len - original_len
    
        # Symmetric padding (avoid edge effects)
        padded_flux = np.pad(flux, (0, pad_width), mode='symmetric')
    
        # Wavelet decomposition and thresholding 
        coeffs = pywt.wavedec(padded_flux, 'db4', mode='per', level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # MAD estimator
        threshold = sigma * np.sqrt(2 * np.log(len(padded_flux)))
        denoised_coeffs = [pywt.threshold(c, threshold) for c in coeffs]
    
        # Reconstruct and truncate
        denoised_padded = pywt.waverec(denoised_coeffs, 'db4', mode='per')
        denoised_flux = denoised_padded[:original_len]
    
        return denoised_flux
    
    def _load_and_process_data(self, fits_path):
        """Load SDSS spectral FITS and extract key parameters"""
        if not os.path.exists(fits_path):
            raise FileNotFoundError(f"File {fits_path} doesn't exist")
    
        with fits.open(fits_path) as hdul:
            # Extract data from the first HDU
            data = hdul[1].data
            wl = 10**data['loglam']  # logarithmic wavelength to linear (Unit: Ångströms)
            flux = data['flux']              # Flux value（Unit：10⁻17 erg/cm²/s/Å）
        
            # Extract metadata from the second HDU
            metadata = hdul[2].data
        
            # Data cleaning
            normalized_flux = self.data_cleaning(flux)
            valid_mask = (flux > 0) & np.isfinite(flux)
            clean_wl = wl[valid_mask]
        
            # Data validation after initial cleaning
            if len(clean_wl) != len(normalized_flux):
                raise ValueError("Wavelength and flux lengths do not match after cleaning") 
        
            # Wavelet denoising 
            try:
                denoised_flux = self.wavelet_denoising(normalized_flux)
            except ValueError as e:
                print(f"Fail to denoise: {str(e)}, using normalized flux instead")
                denoised_flux = normalized_flux
        
            # Final length validation
            if len(denoised_flux) != len(clean_wl):
                raise RuntimeError(f"Abnormal length after denoising (input{len(clean_wl)}，output{len(denoised_flux)}）")
            assert len(clean_wl) == len(denoised_flux)

            return {
                'wavelength': clean_wl,
                'flux': denoised_flux,
                'z': metadata['Z'][0],         # redshift
                'teff': metadata['ELODIE_TEFF'][0],          # effective temperature（K）
                'feh': metadata['ELODIE_FEH'][0],            # metallity[Fe/H]
                'logg': metadata['ELODIE_LOGG'][0],           # surface gravity（log cm/s²）
                'snr': np.median(denoised_flux) / np.std(denoised_flux),  # signal-to-noise ratio
                'emission_lines': self._detect_emission_lines(clean_wl, denoised_flux)  # emission lines
            }

    def _detect_emission_lines(self, wl, flux):
        """Detect major emission lines"""
        peaks = []
        for line in [6563, 4861, 4340]:  # Hα, Hβ, Hγ
            # Correctly find index using numpy's argmin
            idx = np.argmin(np.abs(wl - line))
            if flux[idx] > np.percentile(flux, 90):
                peaks.append(('H' + str(line), flux[idx]))  # Simplified name generation logic
        return peaks

    def _apply_global_settings(self):
        """Ensure scale object creation"""
        """Apply global music parameters"""
        # Select scale based on [Fe/H]
        try:
            scale_mode = PARAM_MAPPING['feh']['mapping'](self.data['feh'])
            self.scale = {
                'minor': scale.HarmonicMinorScale('C4'),
                'lydian': scale.LydianScale('G4')
            }.get(scale_mode, scale.MajorScale('C4'))  # Add default case
            
            # Add closestPitch method
            self.scale.closestPitch = lambda p: min(
                self.scale.getPitches(),
                key=lambda x: abs(x.midi - p.midi))
        except Exception as e:
            logging.error(f"Fail to initialize scale: {str(e)}")
            self.scale = scale.MajorScale('C4')
            self.scale.closestPitch = lambda p: p  # Simple fault tolerance
        
        transpose, tempo_factor = PARAM_MAPPING['z']['mapping'](self.data['z'])
        self.score.insert(0, tempo.MetronomeMark(
            number=80 * tempo_factor))
        self.transpose = transpose

        pass

    def generate_full_composition(self):
        """Generate complete composition"""
        # Harmony layer
        self._add_harmony_layer()
        
        # Melody layer
        self._add_melody_layer()
        
        # Rhythm layer
        self._add_rhythm_layer()
        
        # Bass layer
        self._add_bass_layer()
        
        # Post-processing
        self._apply_post_processing()
        
        return self.score

    def generate_track(self, data, instr, track_type='melody'):
        """
        Generate single instrument track
        :param data: Dictionary containing wavelength and flux
        :param instr: music21 instrument object
        :param track_type: melody/rhythm
        """
        s = stream.Part()
        s.append(instr)
    
        wl = data['wavelength']
        flux = data['flux']
        flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
    
        assert len(wl) == len(flux)

        if track_type == 'melody':
            # Main melody generation:
            mask = (wl > 4000) & (wl < 8000)
            # Apply mask synchronously
            wl_masked = wl[mask]
            flux_masked = flux[mask]
    
            # Verify again
            assert len(wl_masked) == len(flux_masked), "Unequal lengths after masking"
    
            # Downsampling (keep synchronized)
            step = 10
            wl_sampled = wl_masked[::step]
            flux_sampled = flux_masked[::step]

            for w, f in zip(wl_sampled, flux_sampled):  
                midi_num = 60 + int((w - 6500) / 100 * 12)
                n = note.Note(midi_num)
                n.duration.quarterLength = 0.25 + f * 2
                n.volume.velocity = int(f * 127)
                s.append(n)
            
        elif track_type == 'rhythm':
            # Rhythm generation: percussion based on flux variance
            window_size = 50
            variances = [np.var(flux_norm[i:i+window_size]) 
                        for i in range(0, len(flux_norm), window_size)]
            for var in variances:
                if var > 0.1:
                    s.append(note.Note('D2', quarterLength=0.25)) 
                    s.insert(0, SnareDrum()) 
                elif var > 0.05:
                    s.append(note.Rest(quarterLength=0.25))
    
        return s
    
    def generate_melody(self, data, instr):
        """Main melody generation(package method)"""
        s = stream.Part()
        s.append(instr) 
        return self.generate_track(data, instr, 'melody')

    def generate_harmony(self, wavelength, flux):
        """Generate dynamic chords based on spectral features"""
        chord_progression = []
    
        # Local flux peaks as chord change points
        peak_indices = np.where(np.diff(np.sign(np.diff(flux))) < 0)[0] + 1
    
        for i in range(0, len(peak_indices)-1):
            segment = flux[peak_indices[i]:peak_indices[i+1]]
            wl_segment = wavelength[peak_indices[i]:peak_indices[i+1]]
        
            # Extract traits
            flux_var = np.var(segment)
            wl_skew = np.mean(wl_segment) / 1000  # to visible light range
        
            # Choose logic for chord
            if flux_var > 0.1:
                root = pitch.Pitch(int(60 + (wl_skew % 12)))
                chord_type = 'dominant7' if wl_skew > 6 else 'minor7'
            else:
                root = pitch.Pitch(60 + int(len(segment)/100))
                chord_type = 'major'
        
            # Add tension tones
            extensions = ['9', '11'] if flux_var > 0.05 else []
            chord_obj = harmony.ChordSymbol(root=root, kind=chord_type, 
                                        extensions=extensions)
            chord_progression.append(chord_obj)
    
        return chord_progression

    def generate_rhythm(self, flux, snr):
        rhythm_stream = stream.Part()
        rhythm_stream.append(instrument.SnareDrum())

        """Generate compound rhythm based on flux SNR"""
        flux_diff = np.diff(flux)
        flux_norm = (flux_diff - np.min(flux_diff)) / (np.max(flux_diff) - np.min(flux_diff))
    
        rhythm_stream = stream.Part()
        rhythm_stream.append(instrument.SnareDrum())
    
        # Base rhythm pattern: use quarterLength values directly
        basic_rhythm = [0.25, 1.0, 0.5]  
        # Add complexity based on SNR
        complexity = int(snr / 10)  
    
        for i in range(len(flux_norm)):
            prob = flux_norm[i]
            if np.random.rand() < prob**2:  
                dur = basic_rhythm[np.random.choice(len(basic_rhythm))]
                if complexity > 3:
                    dur = dur * 2/3
                snare = note.Note('D2', quarterLength=dur)
                rhythm_stream.append(snare)
            else:
                rhythm_stream.append(note.Rest(quarterLength=0.25))
    
        # Add random fills
        if complexity > 2:
            fill = stream.Measure()
            fill.append(note.Note('A2', quarterLength=0.125))
            fill.append(note.Note('C3', quarterLength=0.125))
            rhythm_stream.insert(np.random.randint(4, len(rhythm_stream)), fill)
    
        return rhythm_stream
    
    def _segment_spectrum(self):
        """Segment spectral data"""
        # Simple segmentation: 100 data points per segment
        segment_size = 100
        segments = []
        for i in range(0, len(self.data['flux']), segment_size):
            segment = {
                'wl': self.data['wavelength'][i:i+segment_size],
                'flux': self.data['flux'][i:i+segment_size]
            }
            segments.append(segment)
        return segments
    
    def _add_harmony_layer(self):
        """Intelligent harmony generation"""
        chord_prog = []
        for seg in self._segment_spectrum():
            # Dynamic chord generation logic
            tension = PARAM_MAPPING['wavelength']['mapping'](seg['wl'])[0]
            chord_type = 'maj7' if tension > 1.5 else 'min7'
            root = self.scale.getPitches()[int(len(seg['flux'])%7)]
            
            # Add extended tones
            extensions = ['9', '11'] if tension > 2 else []
            chord_prog.append(harmony.ChordSymbol(
                root=root.transpose(self.transpose),
                kind=chord_type,
                extensions=extensions
            ))
        
        harmony_part = stream.Part(instrument.Piano())
        harmony_part.append(chord_prog)
        self.score.append(harmony_part)

    def _add_melody_layer(self):
        """Decorative melody generation"""
        melody = self.generate_melody(
            self.data, 
            instrument.Violin()
        ).transpose(self.transpose)
        
        decorated_melody = self.decorate_melody(melody)
        
        # Apply dynamic expressions
        self._apply_dynamics(decorated_melody, self.data['flux'])
        
        self.score.append(decorated_melody)

    def _apply_dynamics(self, melody_stream, flux):
        """Apply dynamic expressions"""
        try:
            flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
            flux_norm = np.asarray(flux_norm).flatten() 
            notes = list(melody_stream.flatten().notes)
            if not notes:
                return
            
            step = max(len(flux_norm) // len(notes), 1)
            flux_sampled = flux_norm[::step]
        
            if len(flux_sampled) < len(notes):
                flux_sampled = np.tile(flux_sampled, (len(notes) // len(flux_sampled) + 1))[:len(notes)] 

            for n, f in zip(notes, flux_sampled):
                f_scalar = float(f)
                n.volume.velocity = int(70 + 50 * f_scalar)
                if f_scalar > 0.7:
                    n.expressions.append(dynamics.Crescendo())
                elif f_scalar < 0.3:
                    n.expressions.append(dynamics.Diminuendo())
        except Exception as e:
            logging.error(f"Fail to apply dynamics: {str(e)}")

    def _add_rhythm_layer(self):
        """Compound rhythm generation"""
        rhythm_part = self.generate_rhythm(
            self.data['flux'], 
            self.data['snr']
        ).transpose(self.transpose)
        
        #  Adjust percussion intensity based on emission line strength
        for n in rhythm_part.flatten().notes:
            if isinstance(n, note.Note):
                line_strength = sum([v for k,v in self.data['emission_lines']])
                n.volume.velocity = int(100 * line_strength)
        
        self.score.append(rhythm_part)

    def _add_bass_layer(self):
        """Walking bass generation"""
        bass_part = self._generate_metallicity_bass()
        if not bass_part.notes:
            bass_part.append(note.Rest(quarterLength=4))  # Add whole rest if no notes

        self.score.append(bass_part)
        
    def _apply_post_processing(self):
        """Post-effect processing"""
        
        self._spatial_panning()
        self._add_reverb()
        self._apply_spatial_effects()
        self._dynamic_shaping()

    def _apply_spatial_effects(self):
        """3D sound field processing"""
        for part in self.score.parts:
            instr = part.getInstrument()
            instr.pan = self.spatial_pan
            instr.midiProgram = 19 + int(self.reverb_strength * 10)

    def _generate_metallicity_bass(self):
        """Generate bass based on metallicity"""
        
        # Get base pitch
        try:
            base_pitch = self.key.tonic.transpose(-24)
        except AttributeError:
            base_pitch = pitch.Pitch('C2')

        if not hasattr(self, 'key'):
            self.key = key.Key('C')  
        if not hasattr(self, 'spatial_pan'):
            self.spatial_pan = 0.0


        bass_stream = stream.Part()
        bass_stream.append(instrument.AcousticBass())

        feh = self.data['feh']
    
        if feh >= 0:  # Jazz-style bass for high metallicity
            # Get chord root (based on wavelength statistics)
            root_pitch = self.scale.getPitches()[int(np.mean(self.data['wavelength']) % 7)]

            # Create walking bass pattern
            for i in range(0, len(self.data['flux']), 50): 
                extensions = []
                if feh > 0.5:
                    extensions = ['7', '9']
                elif feh > 0.2:
                    extensions = ['7']

                current_chord = harmony.ChordSymbol(
                    root=root_pitch.transpose(-12), 
                    kind='dominant' if feh > 0 else 'major',
                    extensions=extensions
                )

                # Add rhythmic variation based on flux variance
                flux_segment = self.data['flux'][i:i+50]
                duration = 0.5 + (np.var(flux_segment)/0.1)  
                current_chord.duration.quarterLength = min(duration, 2.0)

                bass_stream.append(current_chord)

                root_pitch = self.scale.nextPitch(root_pitch)

        else:  # Minimalist bass for low metallicity
            base_pitch = self.key.tonic.transpose(-24)  

            # Generate pulse bass based on flux peaks
            peak_indices = np.where(np.diff(np.sign(np.diff(self.data['flux']))) < 0)[0]
            for idx in peak_indices[::10]: 
                n = note.Note(base_pitch)
                n.duration.quarterLength = 0.25
                # Apply velocity mapping from flux values
                n.volume.velocity = int(40 + 80 * (self.data['flux'][idx]/np.max(self.data['flux'])))
                bass_stream.append(n)
                # Add rests to maintain rhythm
                bass_stream.append(note.Rest(quarterLength=0.75))

        self._smooth_bass_line(bass_stream)
        return bass_stream
    
    def _smooth_bass_line(self, bass_stream):
        """Smooth bass line (avoid large leaps)"""
        prev_pitch = None
        for element in bass_stream:
            if isinstance(element, (note.Note, chord.Chord)):
                current_pitch = element.root() if isinstance(element, chord.Chord) else element.pitch
                if prev_pitch:
                    # Limit interval to perfect fourth
                    interval_obj = interval.Interval(prev_pitch, current_pitch)
                    if interval_obj.semitones > 5:
                        new_pitch = prev_pitch.transpose(-3)
                        if isinstance(element, chord.Chord):
                            element.root(new_pitch)
                        else:
                            element.pitch = new_pitch
                prev_pitch = current_pitch
  
    # --------------------------
    # Music enhancement methods
    # --------------------------
  
    def decorate_melody(self, melody_stream):
        """Add musical decorations to melody"""
        # Parameterized decoration rules
        DECORATION_PROBS = {
            'trill': 0.3,
            'grace_note': 0.4,
            'slide': 0.2,
            'vibrato': 0.6
        }
        for n in melody_stream.flatten().notes:
            
            if np.random.rand() < DECORATION_PROBS['grace_note']:
                grace = note.Note(n.pitch.midi - 1)
                grace.duration.quarterLength = 0.1
                melody_stream.append(grace)
            
        return melody_stream
    
    def _spatial_panning(self):
        """Stereo panning distribution"""
        for i, part in enumerate(self.score.parts):
            pan_value = -0.5 + i*0.3  

        # Ensure track contains instrument object
        if not part.getElementsByClass(instrument.Instrument):
            part.insert(0, instrument.Piano())  
        
        
        instr = part.getElementsByClass(instrument.Instrument)[0]
        instr.pan = pan_value

    def _add_reverb(self):
        """Reverb related to redshift (implement via timbre selection)"""
        reverb_strength = 1.0 + self.data['z'] * 0.5  
        for part in self.score.parts:
            if not part.getElementsByClass(instrument.Instrument):
                part.insert(0, instrument.Piano())
            instr = part.getElementsByClass(instrument.Instrument)[0]
            
            instr.midiProgram = int(19 * reverb_strength)  

    def _dynamic_shaping(self):
        """Dynamic shaping based on flux curve"""
        try:
            flux_norm = (self.data['flux'] - np.min(self.data['flux'])) / \
                    (np.max(self.data['flux']) - np.min(self.data['flux']))
            dynamic_levels = ['pp', 'p', 'mp', 'mf', 'f', 'ff']
            dynamic_volumes = {'pp': 30, 'p': 50, 'mp': 70, 'mf': 90, 'f': 110, 'ff': 127}
            for n, f in zip(self.score.flatten().notes, flux_norm):
                dynamic_idx = min(int(f * len(dynamic_levels)), len(dynamic_levels) - 1)
                dynamic = dynamics.Dynamic(dynamic_levels[dynamic_idx])
                n.addLyric(dynamic.value) 
                n.volume.velocity = dynamic_volumes[dynamic_levels[dynamic_idx]]  
        except Exception as e:
            logging.error(f"Fail to dynamic shaping: {str(e)}")

# Add upgrade content in AstroMusicGenerator.py
from music21.analysis import discrete
from music21 import environment
env = environment.Environment()
env['musicxmlPath'] = '/usr/local/bin/musescore'    # Optional: Set MuseScore path
env['autoDownload'] = 'allow'                       # Allow automatic downloading of necessary components

class EnhancedAstroMusicGenerator(AstroMusicGenerator):
    def __init__(self, fits_path):
        super().__init__(fits_path)
        try:
            self._advanced_mappings() 
        except Exception as e:
            logging.warning(f"Fail to advanced mapping: {str(e)}")
            
            self.key = key.Key('C')
            self.spatial_pan = 0.0
            self.reverb_strength = 1.0
    
    def _advanced_mappings(self):
        """Enhanced parameter mapping system"""
        
        self._analyze_scale()
        
        self.primary_instrument = self._select_primary_instrument()
        self.secondary_instrument = self._select_secondary_instrument()
        
        self.spatial_pan = self.data.get('RA', 0) / 180.0 - 1.0  
        self.reverb_strength = min(self.data.get('Dec', 0) / 90.0, 1.0)

    def _analyze_scale(self):
        """Scale analysis (with exception handling)"""
        try:
            dummy_melody = self._generate_base_melody()
            analyzer = discrete.KrumhanslSchmuckler()
            analyzed_key = analyzer.getSolution(dummy_melody)
            # Validate analysis results
            if analyzed_key is not None:
                self.key = analyzed_key
            else:
                self.key = key.Key('C')  # Default to C major
        except Exception as e:
            logging.warning(f"Fail: {str(e)}, default to C major")
            self.key = key.Key('C')

    def _select_primary_instrument(self):
        """Assign primary instrument based on emission lines"""
        line_strengths = {k:v for k,v in self.data['emission_lines']}
        if line_strengths.get('Hα', 0) > 500:
            return instrument.Violin()
        elif line_strengths.get('CaII', 0) > 300:
            return instrument.Trumpet()
        elif line_strengths.get('OIII', 0) > 200:
            return instrument.Flute()
        return instrument.Piano()
    
    def _select_secondary_instrument(self):
        """Select secondary instrument based on logg"""
        if self.data['logg'] > 4.0:
            return instrument.Marimba()
        return instrument.Contrabass()

    def generate_full_composition(self):
        """Enhanced full composition generation"""
        # Main melody layer with redshift transposition
        melody = self._generate_dynamic_melody().transpose(int(self.data['z']*20))
        
        # Harmony layer (avoid parallel fifths)
        harmony = self._generate_harmonic_progression()
        
        # Bass layer driven by metallicity
        bass = self._generate_metallicity_bass()
        
        # Spatial effects
        self._apply_spatial_effects()
        
        return self.score

    def _generate_dynamic_melody(self):
        """Generate dynamic melody (flux-linked velocity)"""
        melody_stream = stream.Part()
        for w, f in zip(self.data['wavelength'], self.data['flux']):
            f = np.nan_to_num(f, nan=0.5)
            midi_num = 60 + int((w - 5000) / 100 * 12)  
            velocity = int(40 + 87 * (f ** 0.5)) if not np.isnan(f) else 64
            
            n = note.Note(midi_num)
            n.duration.quarterLength = 0.25 + (f / np.max(self.data['flux'])) * 2
            n.volume.velocity = velocity
            melody_stream.append(n)
        return melody_stream

    def _generate_harmonic_progression(self):
        """Smart harmony generation (with music theory validation)"""
        # Check at method start
        if not hasattr(self, 'scale'):
            self._apply_global_settings()

        chord_prog = []
        prev_chord = None
        scale_pitches = [p.midi for p in self.scale.getPitches()]  

        def find_closest_pitch(target_pitch):
            scale_pitches = [p.midi for p in self.scale.getPitches()]
            diffs = np.abs(np.array(scale_pitches) - target_pitch.midi)
            return self.scale.getPitches()[np.argmin(diffs)]

        for seg in self._segment_spectrum():
            chord = self._create_smart_chord(seg)
            if prev_chord and not self._check_harmonic_rules(prev_chord, chord):
                chord = self._adjust_chord(prev_chord, chord)
            
            if chord.root().midi not in scale_pitches: 
                new_root = find_closest_pitch(chord.root())
                chord.root(new_root)
            chord_prog.append(chord)
            prev_chord = chord
        return chord_prog

    def _check_harmonic_rules(self, chord1, chord2):
        """Music theory rule validation"""
        """Use new harmony analysis interface"""
        from music21.analysis import discrete
        analysis_result = discrete.analyzeStream([chord1, chord2], 'parallelFifths')
        return analysis_result.resultValue < 1 
    
    def _apply_spatial_effects(self):
        """3D sound field processing"""
        for part in self.score.parts:
            instr = part.getInstrument()
            instr.pan = self.spatial_pan
            instr.midiProgram = 19 + int(self.reverb_strength * 10)
    
    def _generate_base_melody(self):
        """Generate base melody for scale analysis"""
        base_stream = stream.Stream()
        for w in np.linspace(4000, 8000, num=20):
            midi_num = 60 + int((w - 5000) / 100 * 12)
            base_stream.append(note.Note(midi_num))
        return base_stream

    def _create_smart_chord(self, seg):
        """Smart chord generation"""
        root = self.scale.getPitches()[int(len(seg['flux']) % 7)]
        third = root.transpose(4)
        fifth = root.transpose(7)
        # Add extensions based on flux variance
        flux_var = np.var(seg['flux'])
        extensions = []
        if flux_var > 0.1:
            extensions = [7, 9] if self.data['feh'] > 0 else [7]
        return chord.Chord([root, third, fifth] + extensions)

    def _adjust_chord(self, prev_chord, current_chord):
        """Harmony correction (example)"""
        return current_chord.transpose(1)  # Shift up a step to avoid parallel fifths
    
