import mido
import time
import os
import numpy as np
try:
    import pygame
except ImportError:
    pygame = None

# Pentatonic Scale: C, D, E, G, A
# Expanded for smoother transitions and wider range
PENTATONIC_SCALE = [
    36, 38, 40, 43, 45,  # C2 - A2 (Bass)
    48, 50, 52, 55, 57,  # C3 - A3
    60, 62, 64, 67, 69,  # C4 - A4 (Middle)
    72, 74, 76, 79, 81,  # C5 - A5
    84, 86, 88, 91, 93,  # C6 - A6
    96                   # C7
]

def map_value_to_note(value, min_val, max_val, last_note=None):
    """
    Maps value to note, with an option to keep it close to the last note 
    to avoid large melodic jumps (smoothness).
    """
    if max_val == min_val:
        return 60 # Middle C
        
    # Normalize 0-1
    normalized = (value - min_val) / (max_val - min_val)
    scale_len = len(PENTATONIC_SCALE)
    
    # Base index calculation
    target_idx = int(normalized * (scale_len - 1))
    
    # Smoothing logic: if we have a last note, try to pick the scale note closest to it
    # but also influenced by the new value.
    # Actually, simpler approach for "pleasant":
    # 1. Map value primarily to pitch height
    # 2. But constrain the range to "singable" range (e.g. C3 to C6)
    
    # Let's focus on C4 (60) to C6 (84) for melody
    # Indices in PENTATONIC_SCALE: 
    # 60 is index 10. 84 is index 20.
    # Let's map normalized 0-1 to index 10-22
    
    min_idx = 10
    max_idx = 22
    mapped_idx = int(min_idx + normalized * (max_idx - min_idx))
    
    # Ensure bounds
    mapped_idx = max(0, min(mapped_idx, scale_len - 1))
    
    # If we wanted to smooth jumps, we could average with last index?
    # For now, let's trust the restricted range to reduce craziness.
    
    return PENTATONIC_SCALE[mapped_idx]

def generate_full_arrangement(times, values, rhythm_data, filename='chemical_full_song.mid'):
    """
    Generate elegant piano arrangement.
    Removes drums for a cleaner, more classical feel.
    """
    mid = mido.MidiFile(type=1)
    
    # 1. Setup Timing & Global Config
    bpm = rhythm_data.get('bpm', 120)
    # Slow down for elegance
    if bpm > 100: bpm = bpm * 0.8
    if bpm < 60: bpm = 60
    
    mid.ticks_per_beat = 480
    ticks_per_second = (bpm / 60) * 480
    
    total_duration = times[-1] - times[0] if times else 0
    start_time = times[0]
    
    # --- Track 0: Melody (Acoustic Grand Piano) ---
    track0 = mido.MidiTrack()
    mid.tracks.append(track0)
    track0.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    track0.append(mido.MetaMessage('track_name', name='Piano Melody', time=0))
    # Program 0: Acoustic Grand Piano
    track0.append(mido.Message('program_change', program=0, time=0, channel=0))

    # --- Track 1: Arpeggio/Harmony (Piano Left Hand) ---
    track1 = mido.MidiTrack()
    mid.tracks.append(track1)
    track1.append(mido.MetaMessage('track_name', name='Piano Harmony', time=0))
    # Same instrument
    track1.append(mido.Message('program_change', program=0, time=0, channel=1))
    
    # --- Generate Melody Content ---
    melody_events = []
    min_val = np.min(values)
    max_val = np.max(values)
    
    last_note = None
    
    # Use 1/8 note quantization
    seconds_per_beat = 60 / bpm
    seconds_per_unit = seconds_per_beat / 2 # 8th notes
    
    # Quantize input times
    quantized_notes = {}
    
    for i, t in enumerate(times):
        rel_time = t - start_time
        grid_slot = round(rel_time / seconds_per_unit)
        tick = int(grid_slot * (480/2))
        
        if tick not in quantized_notes:
            quantized_notes[tick] = []
        quantized_notes[tick].append(values[i])
        
    sorted_ticks = sorted(quantized_notes.keys())
    
    for tick in sorted_ticks:
        vals = quantized_notes[tick]
        avg_val = sum(vals) / len(vals)
        
        # Use simple mapping
        note = map_value_to_note(avg_val, min_val, max_val, last_note)
        
        # Humanize velocity
        vel = np.random.randint(60, 90) # Gentle range
        
        # Duration: slightly less than full 8th
        dur_tick = int(480/2 * 0.9)
        
        melody_events.append({'tick': tick, 'type': 'note_on', 'note': note, 'velocity': vel, 'channel': 0})
        melody_events.append({'tick': tick + dur_tick, 'type': 'note_off', 'note': note, 'velocity': 0, 'channel': 0})
        last_note = note
    
    # Write Melody Track
    melody_events.sort(key=lambda x: x['tick'])
    last_tick = 0
    for e in melody_events:
        delta = max(0, e['tick'] - last_tick)
        track0.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta, channel=0))
        last_tick = e['tick']

    # --- Generate Harmony (Arpeggios) ---
    # Gentle left hand patterns based on implied harmony
    harmony_events = []
    ticks_per_bar = 480 * 4
    total_bars = int((sorted_ticks[-1] if sorted_ticks else 0) / ticks_per_bar) + 2
    
    # Simple Alberti bass or broken chords
    progression = [
        [48, 55, 60], # C3
        [45, 52, 57], # A2
        [41, 48, 53], # F2
        [43, 50, 55]  # G2
    ]
    
    for bar in range(total_bars):
        chord = progression[bar % 4]
        base_tick = bar * ticks_per_bar
        
        # Play broken chord pattern: Root - 5th - 3rd - 5th (Eighth notes)
        pattern_notes = [chord[0], chord[2], chord[1], chord[2]] * 2 # 8 notes for 4/4 bar
        
        for i, note in enumerate(pattern_notes):
            tick = base_tick + i * 240 # 8th note spacing
            vel = 55 + (5 if i % 4 == 0 else 0) # Accent downbeat slightly
            
            harmony_events.append({'tick': tick, 'type': 'note_on', 'note': note, 'velocity': vel, 'channel': 1})
            harmony_events.append({'tick': tick+220, 'type': 'note_off', 'note': note, 'velocity': 0, 'channel': 1})

    harmony_events.sort(key=lambda x: x['tick'])
    last_tick = 0
    for e in harmony_events:
        delta = max(0, e['tick'] - last_tick)
        track1.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta, channel=1))
        last_tick = e['tick']

    # No drums track anymore
        
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
