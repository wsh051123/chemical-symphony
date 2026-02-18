import mido
import time
import os
import numpy as np
try:
    import pygame
except ImportError:
    pygame = None

# Expanded for smoother transitions and wider range
# Use C Major Scale (Ionian) for a brighter, more "classical" feel
# C, D, E, F, G, A, B
C_MAJOR_SCALE = [
    48, 50, 52, 53, 55, 57, 59,  # C3 - B3 (Low-Mid)
    60, 62, 64, 65, 67, 69, 71,  # C4 - B4 (Middle)
    72, 74, 76, 77, 79, 81, 83,  # C5 - B5 (High)
    84, 86, 88, 89, 91, 93, 95,  # C6 - B6 (Very High - "Crisp")
    96, 98, 100, 101, 103, 105, 107 # C7 - B7 (Brilliant)
]

def map_value_to_note(value, min_val, max_val, last_note=None):
    """
    Maps value to note, favoring higher registers for a crisp sound.
    """
    if max_val == min_val:
        return 72 # C5
        
    # Normalize 0-1
    normalized = (value - min_val) / (max_val - min_val)
    scale_len = len(C_MAJOR_SCALE)
    
    # Focus on Middle-High to High registers for "Likje Lang Lang" crispness
    # Range indices: Let's use roughly index 12 (G4) to 30 (E7)
    min_idx = 10
    max_idx = scale_len - 5 # Leave slight headroom
    
    mapped_idx = int(min_idx + normalized * (max_idx - min_idx))
    
    # Ensure bounds
    mapped_idx = max(0, min(mapped_idx, scale_len - 1))
    
    # If we wanted to smooth jumps, we could average with last index?
    # For now, let's trust the restricted range to reduce craziness.
    
    return C_MAJOR_SCALE[mapped_idx]

def generate_full_arrangement(times, values, rhythm_data, filename='chemical_full_song.mid', style='default'):
    """
    Generate arrangement based on style.
    styles: 'default' (Piano), 'hakimi' (High energy, repetitive, fun)
    """
    mid = mido.MidiFile(type=1)
    
    # 1. Setup Timing & Global Config
    if style == 'hakimi':
        target_bpm = 145 # Fast, energetic (Chipi Chipi style)
    else:
        bpm = rhythm_data.get('bpm', 120)
        target_bpm = 96 # Classical/Piano feel

    mid.ticks_per_beat = 480
    ticks_per_second = (target_bpm / 60) * 480
    
    start_time = times[0]
    total_duration = times[-1] - start_time
    
    # --- Track 0: Melody ---
    track0 = mido.MidiTrack()
    mid.tracks.append(track0)
    track0.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(int(target_bpm)), time=0))
    track0.append(mido.MetaMessage('track_name', name='Melody', time=0))
    
    if style == 'hakimi':
        # Bright Synth or synth lead
        track0.append(mido.Message('program_change', program=80, time=0, channel=0)) # 80: Lead 1 (square) or 81 (sawtooth)
    else:
        # Acoustic Grand Piano
        track0.append(mido.Message('program_change', program=0, time=0, channel=0))

    # --- Track 1: Backing/Harmony ---
    track1 = mido.MidiTrack()
    mid.tracks.append(track1)
    track1.append(mido.MetaMessage('track_name', name='Harmony', time=0))
    
    if style == 'hakimi':
        # Electric Bass or Synth Bass
        track1.append(mido.Message('program_change', program=38, time=0, channel=1)) # 38: Synth Bass 1
    else:
        # Piano Left Hand
        track1.append(mido.Message('program_change', program=0, time=0, channel=1))
    
    # --- Analyze Data Trends ---
    # Convert inputs to numpy for vectorized ops
    time_arr = np.array(times)
    val_arr = np.array(values)
    min_val = np.min(val_arr)
    max_val = np.max(val_arr)
    span_val = max_val - min_val if max_val != min_val else 1.0
    
    # Normalize values 0-1
    norm_vals = (val_arr - min_val) / span_val
    
    # Calculate local gradient (rate of change)
    # We smooth it slightly to avoid jitter
    window_size = max(1, len(val_arr) // 20)
    # Simple rolling average of absolute difference
    # Or just gradient of smoothed signal
    
    # Let's assess "volatility" per musical measure (Bar)
    # Assume 4/4 time signature
    seconds_per_bar = (60 / target_bpm) * 4
    num_bars = int(np.ceil(total_duration / seconds_per_bar)) + 1
    
    melody_events = []
    harmony_events = []
    
    # Global average change to determine thresholds
    global_grad = np.mean(np.abs(np.gradient(norm_vals))) if len(norm_vals) > 1 else 0
    
    last_note = None

    # Progression: I - vi - IV - V (C - Am - F - G)
    chord_roots = [48, 45, 41, 43] # C2, A1, F1, G1
    progression_len = len(chord_roots)


    for bar in range(num_bars):
        bar_start_sec = start_time + bar * seconds_per_bar
        bar_end_sec = bar_start_sec + seconds_per_bar
        
        # Determine bar style based on data in this timeframe
        mask = (time_arr >= bar_start_sec) & (time_arr < bar_end_sec)
        indices = np.where(mask)[0]
        
        # Default to calm if no data (e.g. end of song)
        if len(indices) == 0:
            intensity = 0.0
            avg_height = 0.5
        else:
            local_grad = np.mean(np.abs(np.gradient(norm_vals[indices]))) if len(indices) > 1 else 0
            avg_height = np.mean(norm_vals[indices])
            
            # Normalize intensity relative to global
            if global_grad > 0:
                intensity = local_grad / global_grad
            else:
                intensity = 1.0 # Flat line
        
        # Determine Rhythmic Density (Notes per beat)
        if style == 'hakimi':
            # Hakimi is always high energy
            notes_per_beat = 4 if intensity > 0.8 else 2
            style_label = 'rapid'
            base_velocity = 100 # Loud
        else:
            if intensity > 1.5:
                notes_per_beat = 4
                style_label = 'rapid'
            elif intensity > 0.6:
                notes_per_beat = 2
                style_label = 'flowing'
            else:
                notes_per_beat = 1
                style_label = 'calm'
            base_velocity = 80 + int(avg_height * 35)
            
        # Refine style by height: High value = louder, brighter
        # Increase base velocity for "crisp" piano sound (Lang Lang style: clear articulation)
        # base_velocity calculated above
        
        # --- Generate Melody for this Bar ---
        total_beats = 4
        notes_in_bar = total_beats * notes_per_beat
        ticks_per_note = int(480 / notes_per_beat)
        
        # To avoid "dragging", ensure note duration is shorter than step
        # Shorter staccato for rapid/flowing to emulate crisp finger work
        if style_label == 'rapid':
            gate = 0.5 if style == 'hakimi' else 0.6 # Very crisp staccato
        elif style_label == 'flowing':
            gate = 0.75 # Non-legato
        else:
            gate = 0.9 # Legato for slow parts
            
        note_dur = int(ticks_per_note * gate)
        
        for i in range(notes_in_bar):
            # Calculate absolute tick
            abs_tick_start = (bar * 480 * 4) + (i * ticks_per_note)
            
            # Find data value at this specific fraction of time
            time_offset = (i / notes_in_bar) * seconds_per_bar
            current_sec = bar_start_sec + time_offset
            
            # Interpolate value
            # Find nearest index
            idx = np.searchsorted(time_arr, current_sec)
            idx = max(0, min(idx, len(val_arr)-1))
            val = val_arr[idx]
            
            # Map Pitch
            # Using new C Major scale
            note = map_value_to_note(val, min_val, max_val, last_note)
            
            # Hakimi Optimization: Force notes to be more "jumpy" or arpeggiated if needed?
            # actually map_value_to_note is fine.
            
            # Add micro-dynamics: accent the first note of beat
            current_velocity = base_velocity
            if i % notes_per_beat == 0:
                current_velocity += 10 # Accent
            
            # Add event
            melody_events.append({
                'tick': abs_tick_start, 
                'type': 'note_on', 'note': note, 'velocity': min(127, current_velocity), 'channel': 0
            })
            melody_events.append({
                'tick': abs_tick_start + note_dur, 
                'type': 'note_off', 'note': note, 'velocity': 0, 'channel': 0
            })
            
            last_note = note

        # --- Generate Left Hand Accompaniment for this Bar ---
        
        root = chord_roots[bar % progression_len]
        
        # Chords logic
        if style == 'hakimi':
             # Simple Major/Minor construction
             # C, G, Am, F
             if bar % 4 == 0: chord_notes = [root, root+4, root+7] # C
             elif bar % 4 == 1: chord_notes = [root, root+4, root+7] # G (using mapped root)
             elif bar % 4 == 2: chord_notes = [root, root+3, root+7] # Am
             else: chord_notes = [root, root+4, root+7] # F
        else:
             if bar % 4 == 0: chord_notes = [root, root+4, root+7] # C
             elif bar % 4 == 1: chord_notes = [root, root+3, root+7] # Am
             elif bar % 4 == 2: chord_notes = [root, root+4, root+7] # F
             else: chord_notes = [root, root+4, root+7] # G

        lh_base_tick = bar * 480 * 4
        
        # Left hand velocity
        lh_velocity = 90 if style == 'hakimi' else 60
        
        if style == 'hakimi':
            # "Chipi Chipi" Bass: Octaves on the beat or pounding 8th notes
            # Let'_labels do a driving bass: Root-Root-Root-Root (Eighth notes)
            for eig in range(8):
                n = root - 12 # Deep bass
                tick = lh_base_tick + eig * 240
                harmony_events.append({'tick': tick, 'type': 'note_on', 'note': n, 'velocity': lh_velocity, 'channel': 1})
                harmony_events.append({'tick': tick + 120, 'type': 'note_off', 'note': n, 'velocity': 0, 'channel': 1})
                
                # Add a "clap" or high hat effect on weak beats? 
                # (Can't easy add percussion channel without channel 9, let's stick to harmony track)
                # Maybe play the chord on the off-beat (2 & 4)?
                if eig % 4 == 2 or eig % 4 == 6: # Beats 2 and 4 approx
                     # Chord stab
                     for cn in chord_notes:
                         harmony_events.append({'tick': tick, 'type': 'note_on', 'note': cn, 'velocity': lh_velocity - 10, 'channel': 1})
                         harmony_events.append({'tick': tick + 100, 'type': 'note_off', 'note': cn, 'velocity': 0, 'channel': 1})

        elif style_label == 'calm':
            # ... (Existing Calm Logic)# A Minor
            chord_notes = [root, root+3, root+7]
        elif bar % 4 == 2: # F Major
            chord_notes = [root, root+4, root+7]
        else: # G Major
            chord_notes = [root, root+4, root+7]

        lh_base_tick = bar * 480 * 4
        
        # Left hand velocity - softer than melody
        lh_velocity = 60 
        
        if style == 'calm':
            # Broken chord waltz or arpeggio (Quarter notes)
            # Root - 5th - 3rd - 5th
            for beat in range(4):
                if beat == 0: n = chord_notes[0] # Root
                elif beat == 1: n = chord_notes[2] # 5th
                elif beat == 2: n = chord_notes[1] # 3rd
                else: n = chord_notes[2] # 5th
                
                tick = lh_base_tick + beat * 480
                harmony_events.append({'tick': tick, 'type': 'note_on', 'note': n, 'velocity': lh_velocity - 10, 'channel': 1})
                harmony_events.append({'tick': tick + 400, 'type': 'note_off', 'note': n, 'velocity': 0, 'channel': 1})
            
        elif style == 'flowing':
            # Alberti Bass (Eighth notes) - "Mozart Style"
            # Pattern: Root - 5th - 3rd - 5th (x2 per bar)
            pattern = [0, 2, 1, 2] * 2
            for eig in range(8):
                n = chord_notes[pattern[eig]]
                # Shift octave up for clarity if needed, but Alberti bass is usually close position
                tick = lh_base_tick + eig * 240
                harmony_events.append({'tick': tick, 'type': 'note_on', 'note': n + 12, 'velocity': lh_velocity, 'channel': 1})
                harmony_events.append({'tick': tick + 200, 'type': 'note_off', 'note': n + 12, 'velocity': 0, 'channel': 1})
                
        else: # rapid
            # Fast Arpeggios or Tremolo (Sixteenth notes)
            # Root - 3rd - 5th - 3rd (x4 per bar)
            pattern = [0, 1, 2, 1] * 4
            for six in range(16):
                n = chord_notes[pattern[six]]
                tick = lh_base_tick + six * 120
                harmony_events.append({'tick': tick, 'type': 'note_on', 'note': n + 12, 'velocity': lh_velocity + 10, 'channel': 1})
                harmony_events.append({'tick': tick + 100, 'type': 'note_off', 'note': n + 12, 'velocity': 0, 'channel': 1})

    # Write events to tracks (sort by time)
    melody_events.sort(key=lambda x: x['tick'])
    last_tick = 0
    for e in melody_events:
        delta = max(0, e['tick'] - last_tick)
        track0.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta, channel=0))
        last_tick = e['tick']
        
    harmony_events.sort(key=lambda x: x['tick'])
    last_tick = 0
    for e in harmony_events:
        delta = max(0, e['tick'] - last_tick)
        track1.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta, channel=1))
        last_tick = e['tick']

    mid.save(filename)
    return filename

def play_midi_file(midi_file):
    if not os.path.exists(midi_file):
        print(f"File not found: {midi_file}")
        return

    # Initialize Pygame Mixer
    try:
        pygame.mixer.stop() # stop previous if any
        pygame.mixer.quit()
    except:
        pass
        
    freq = 44100
    bitsize = -16
    channels = 2
    buffer = 1024
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(0.8)
    
    try:
        clock = pygame.time.Clock()
        pygame.mixer.music.load(midi_file)
        print(f"Playing {midi_file}...")
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            clock.tick(30)
    except Exception as e:
        print(f"Playback error: {e}")

# --- Audio Generation for Web (Streamlit) ---
import scipy.io.wavfile
import io

def note_to_freq(note):
    return 440.0 * (2.0 ** ((note - 69) / 12.0))

def generate_audio_preview(times, values, rhythm_data):
    """
    Generate a simple WAV audio preview for streamlit.
    Synthesizes sine waves for melody. Drums are removed for cleaner sound.
    
    Returns: BytesIO object containing WAV data
    """
    sample_rate = 44100
    
    if not times or len(times) == 0:
        return None
    
    start_time = times[0]
    total_seconds = times[-1] - start_time + 3.0 # Add some tail
    num_samples = int(total_seconds * sample_rate)
    
    # Stereo mix accumulator
    mix = np.zeros(num_samples)
    
    # 1. Melody (Sine Waves)
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Note duration (in seconds)
    note_duration = 0.4 # Longer, more piano-like sustain
    
    # Simple Piano-like overtone structure
    def get_piano_wave(freq, duration, sample_rate):
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        # Fundamental + Harmonics
        w1 = 0.6 * np.sin(2 * np.pi * freq * t)
        w2 = 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        w3 = 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        wave = w1 + w2 + w3
        
        # ADSR Envelope (Piano-like decay)
        attack = int(0.01 * sample_rate)
        decay = int(duration * sample_rate) - attack
        
        env_attack = np.linspace(0, 1, attack)
        env_decay = np.linspace(1, 0, decay) ** 2 # Exponential decay looks better
        
        envelope = np.concatenate([env_attack, env_decay])
        
        if len(envelope) != len(wave):
             envelope = np.resize(envelope, len(wave))
             
        return wave * envelope

    for i, t in enumerate(times):
        val = values[i]
        # Calculate start sample
        onset_time = t - start_time
        start_idx = int(onset_time * sample_rate)
        
        # Determine pitch
        # Use last_note logic if we implemented state, but map_value_to_note is stateless here.
        # Just mapping is fine.
        match_note = map_value_to_note(val, min_val, max_val)
        freq = note_to_freq(match_note)
        
        # Generate wave
        wave = get_piano_wave(freq, note_duration, sample_rate)
        wave_len = len(wave)
        
        if start_idx + wave_len < num_samples:
            mix[start_idx:start_idx+wave_len] += wave
        elif start_idx < num_samples:
            # Truncate
            available = num_samples - start_idx
            mix[start_idx:] += wave[:available]

    # Normalize
    max_amp = np.max(np.abs(mix))
    if max_amp > 0:
        mix = mix / max_amp * 0.9

    # Convert to 16-bit PCM
    audio_data = (mix * 32767).astype(np.int16)
    
    # Write to BytesIO
    wav_file = io.BytesIO()
    scipy.io.wavfile.write(wav_file, sample_rate, audio_data)
    wav_file.seek(0)
    
    return wav_file
    max_amp = np.max(np.abs(mix))
    if max_amp > 0:
        mix = mix / max_amp * 0.9

    # Convert to 16-bit PCM
    audio_data = (mix * 32767).astype(np.int16)
    
    # Write to BytesIO
    wav_file = io.BytesIO()
    scipy.io.wavfile.write(wav_file, sample_rate, audio_data)
    wav_file.seek(0)
    
    return wav_file
