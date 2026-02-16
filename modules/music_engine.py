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
    Generate full song with Melody, Arpeggiator/Chords, and enhanced Drums.
    """
    mid = mido.MidiFile(type=1)
    
    # 1. Setup Timing & Global Config
    bpm = rhythm_data.get('bpm', 120)
    # Clamp BPM
    if bpm < 60: bpm = 80
    if bpm > 160: bpm = 120 
    
    mid.ticks_per_beat = 480
    ticks_per_second = (bpm / 60) * 480
    
    total_duration = times[-1] - times[0] if times else 0
    start_time = times[0]
    
    # --- Track 0: Melody (Electric Piano / Bell) ---
    track0 = mido.MidiTrack()
    mid.tracks.append(track0)
    track0.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    track0.append(mido.MetaMessage('track_name', name='Melody', time=0))
    # Program 4: Electric Piano 1 (Rhodes) - More pleasant than generic piano
    track0.append(mido.Message('program_change', program=4, time=0, channel=0))

    # --- Track 1: Pad/Chords (Warm Pad) ---
    track1 = mido.MidiTrack()
    mid.tracks.append(track1)
    track1.append(mido.MetaMessage('track_name', name='Pad', time=0))
    # Program 89: Warm Pad
    track1.append(mido.Message('program_change', program=89, time=0, channel=1))
    
    # --- Track 9: Drums ---
    track9 = mido.MidiTrack()
    mid.tracks.append(track9)
    track9.append(mido.MetaMessage('track_name', name='Drums', time=0))
    
    # --- Generate Melody Content ---
    melody_events = []
    min_val = np.min(values)
    max_val = np.max(values)
    
    last_note = None
    
    # We want to quantize notes to a grid (e.g. 1/8 notes or 1/16 notes)
    # to make it rhythmically coherent, instead of chaotic time timestamps.
    seconds_per_beat = 60 / bpm
    seconds_per_16th = seconds_per_beat / 4
    
    # Quantize input times to nearest 1/16th note grid
    quantized_notes = {} # {tick: [val1, val2...]}
    
    for i, t in enumerate(times):
        rel_time = t - start_time
        # Round to nearest 16th
        grid_slot = round(rel_time / seconds_per_16th)
        tick = int(grid_slot * seconds_per_beat * 480 / 4)
        
        if tick not in quantized_notes:
            quantized_notes[tick] = []
        quantized_notes[tick].append(values[i])
        
    # Create melody from quantized buckets (averaging values in same bucket)
    sorted_ticks = sorted(quantized_notes.keys())
    
    for tick in sorted_ticks:
        vals = quantized_notes[tick]
        avg_val = sum(vals) / len(vals)
        
        note = map_value_to_note(avg_val, min_val, max_val, last_note)
        # Velocity dynamic based on value (higher = louder)
        vel = 70 + int((avg_val - min_val)/(max_val - min_val + 1e-6) * 30)
        
        # Note duration: 1/8 note usually
        dur_tick = 240 # 480 is quarter note
        
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

    # --- Generate Pad/Background ---
    # Create a simple chord progression C -> Am -> F -> G loop every 4 bars?
    # Or just a drone C major chord that shifts based on data trend?
    # Let's do a simple long drone chord (C Major / A Minor ambiguous)
    pad_events = []
    # Every 4 beats (1 bar), play a chord
    ticks_per_bar = 480 * 4
    total_bars = int((sorted_ticks[-1] if sorted_ticks else 0) / ticks_per_bar) + 2
    
    # Chord progression: Cmaj7 - Am7 - Fmaj7 - G7
    progression = [
        [48, 55, 60, 64], # C3, G3, C4, E4
        [45, 52, 57, 60], # A2, E3, A3, C4
        [41, 48, 53, 57], # F2, C3, F3, A3
        [43, 50, 55, 59]  # G2, D3, G3, B3
    ]
    
    for bar in range(total_bars):
        chord = progression[bar % 4]
        tick_start = bar * ticks_per_bar
        tick_end = tick_start + ticks_per_bar - 10 # slightly shorter than bar
        
        for note in chord:
            pad_events.append({'tick': tick_start, 'type': 'note_on', 'note': note, 'velocity': 50, 'channel': 1})
            pad_events.append({'tick': tick_end, 'type': 'note_off', 'note': note, 'velocity': 0, 'channel': 1})
            
    pad_events.sort(key=lambda x: x['tick'])
    last_tick = 0
    for e in pad_events:
        delta = max(0, e['tick'] - last_tick)
        track1.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta, channel=1))
        last_tick = e['tick']

    # --- Generate Drums ---
    # Standard Loop + Peak Accents
    drum_events = []
    total_quarters = int((sorted_ticks[-1] if sorted_ticks else 0) / 480) + 4
    
    # Basic Beat: Kick on 1, 3; Snare on 2, 4; Hi-hat 8ths
    for q in range(total_quarters):
        tick = q * 480
        
        # Hi-hat (42) every quarter (could be 8th)
        drum_events.append({'tick': tick, 'type': 'note_on', 'note': 42, 'velocity': 40, 'channel': 9})
        drum_events.append({'tick': tick+100, 'type': 'note_off', 'note': 42, 'velocity': 0, 'channel': 9})
        
        # Kick (35) on beat 1 and 3 (0, 2 mod 4)
        if q % 4 in [0, 2]:
             drum_events.append({'tick': tick, 'type': 'note_on', 'note': 35, 'velocity': 90, 'channel': 9})
             drum_events.append({'tick': tick+100, 'type': 'note_off', 'note': 35, 'velocity': 0, 'channel': 9})
             
        # Snare (38) on beat 2 and 4 (1, 3 mod 4)
        if q % 4 in [1, 3]:
             drum_events.append({'tick': tick, 'type': 'note_on', 'note': 38, 'velocity': 80, 'channel': 9})
             drum_events.append({'tick': tick+100, 'type': 'note_off', 'note': 38, 'velocity': 0, 'channel': 9})

    # Accents from Data Peaks
    onsets = rhythm_data.get('onsets', [])
    for t in onsets:
        # Quantize peak to nearest 1/8 note
        rel_time = t - start_time
        grid_slot = round(rel_time / (seconds_per_beat/2))
        tick = int(grid_slot * (480/2))
        
        # Crash Cymbal (49)
        drum_events.append({'tick': tick, 'type': 'note_on', 'note': 49, 'velocity': 100, 'channel': 9})
        drum_events.append({'tick': tick+200, 'type': 'note_off', 'note': 49, 'velocity': 0, 'channel': 9})
        
        # Extra Kick
        drum_events.append({'tick': tick, 'type': 'note_on', 'note': 36, 'velocity': 110, 'channel': 9})
        drum_events.append({'tick': tick+100, 'type': 'note_off', 'note': 36, 'velocity': 0, 'channel': 9})

    drum_events.sort(key=lambda x: x['tick'])
    last_tick = 0
    for e in drum_events:
        delta = max(0, e['tick'] - last_tick)
        track9.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta, channel=9))
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
    Synthesizes sine waves for melody and white noise for drums.
    
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
    
    # Note duration (in seconds) - lets make it slightly staccato
    note_duration = 0.2 
    
    for i, t in enumerate(times):
        val = values[i]
        # Calculate start sample
        onset_time = t - start_time
        start_idx = int(onset_time * sample_rate)
        
        # Determine pitch
        match_note = map_value_to_note(val, min_val, max_val)
        freq = note_to_freq(match_note)
        
        # Generate sine wave for this note
        note_samples = int(note_duration * sample_rate)
        if start_idx + note_samples >= num_samples:
            note_samples = num_samples - start_idx
            
        if note_samples <= 0: continue
            
        t_vec = np.linspace(0, note_duration, note_samples, endpoint=False)
        # Sine wave with simple envelope
        envelope = np.concatenate([
            np.linspace(0, 1, int(0.01*sample_rate)), 
            np.ones(note_samples - int(0.02*sample_rate)), 
            np.linspace(1, 0, int(0.01*sample_rate))
        ])
        # Fix envelope length mismatch due to rounding
        if len(envelope) != len(t_vec):
            envelope = np.ones(len(t_vec)) # Fallback
            
        wave = 0.3 * np.sin(2 * np.pi * freq * t_vec) * envelope
        
        mix[start_idx:start_idx+note_samples] += wave

    # 2. Rhythm (Noise Bursts for Drums)
    bpm = rhythm_data.get('bpm', 120)
    onsets = rhythm_data.get('onsets', [])
    
    # Hi-hat (steady)
    beat_interval = 60 / bpm
    current_t = 0
    while current_t < total_seconds:
        start_idx = int(current_t * sample_rate)
        dur = 0.05 # Short hi-hat
        samps = int(dur * sample_rate)
        
        if start_idx + samps < num_samples:
            noise = np.random.uniform(-0.1, 0.1, samps)
            # High pass filter approximation (just noise is fine for preview)
            mix[start_idx:start_idx+samps] += noise
            
        current_t += beat_interval
        
    # Kick/Crash on peaks
    for t in onsets:
        onset_time = t - start_time
        start_idx = int(onset_time * sample_rate)
        
        # Kick drum (low sine sweep)
        dur = 0.15
        samps = int(dur * sample_rate)
        
        if start_idx + samps < num_samples:
            t_vec = np.linspace(0, dur, samps)
            # Sweep from 150Hz to 50Hz
            freq_sweep = np.linspace(150, 50, samps)
            kick = 0.4 * np.sin(2 * np.pi * freq_sweep * t_vec)
            mix[start_idx:start_idx+samps] += kick

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
