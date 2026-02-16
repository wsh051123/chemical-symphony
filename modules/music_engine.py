import mido
import time
import os
import numpy as np
try:
    import pygame
except ImportError:
    pygame = None

# Pentatonic Scale: C, D, E, G, A
PENTATONIC_SCALE = [
    60, 62, 64, 67, 69,  # C4 - A4
    72, 74, 76, 79, 81,  # C5 - A5
    84, 86, 88, 91, 93   # C6 - A6
]

def map_value_to_note(value, min_val, max_val):
    if max_val == min_val:
        return PENTATONIC_SCALE[0]
    normalized = (value - min_val) / (max_val - min_val)
    scale_len = len(PENTATONIC_SCALE)
    index = int(normalized * (scale_len - 1))
    index = max(0, min(index, scale_len - 1))
    return PENTATONIC_SCALE[index]

def generate_track_from_events(events):
    """
    Helper to convert absolute tick events to relative delta time messages.
    events: list of {'tick': int, 'type': str, 'note': int, 'velocity': int}
    """
    track = mido.MidiTrack()
    events.sort(key=lambda x: x['tick'])
    
    last_tick = 0
    for e in events:
        delta = max(0, e['tick'] - last_tick) # Ensure non-negative
        track.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta))
        last_tick = e['tick']
        
    return track

def generate_full_arrangement(times, values, rhythm_data, filename='chemical_full_song.mid'):
    """
    Generate full song with Melody (from data values) and Rhythm (from data peaks).
    """
    mid = mido.MidiFile(type=1)
    
    # 1. Setup Timing
    bpm = rhythm_data.get('bpm', 120)
    # Clamp BPM to be playable
    if bpm < 40: bpm = 80 # If too slow, double it or set baseline
    if bpm > 200: bpm = 120 
    
    mid.ticks_per_beat = 480
    ticks_per_second = (bpm / 60) * 480
    
    # Tempo track (Track 0 often used for meta events in Type 1)
    # But mido handles tracks simply. Let's put tempo in the first track.
    track0 = mido.MidiTrack()
    mid.tracks.append(track0)
    track0.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    track0.append(mido.MetaMessage('track_name', name='Melody', time=0))
    track0.append(mido.Message('program_change', program=0, time=0, channel=0)) # Piano

    # 2. Melody Events
    melody_events = []
    min_val = np.min(values)
    max_val = np.max(values)
    start_time = times[0]
    
    fixed_duration = 0.25 # seconds per note
    
    for i, t in enumerate(times):
        val = values[i]
        tick_start = int((t - start_time) * ticks_per_second)
        tick_end = tick_start + int(fixed_duration * ticks_per_second)
        
        note = map_value_to_note(val, min_val, max_val)
        vel = 60 + int((val - min_val)/(max_val - min_val + 1e-6) * 40)
        
        melody_events.append({'tick': tick_start, 'type': 'note_on', 'note': note, 'velocity': vel})
        melody_events.append({'tick': tick_end, 'type': 'note_off', 'note': note, 'velocity': 0})
        
    # Add melody events to track 0 (after initial meta events)
    # Actually, let's just make a new list including the meta stuff or handle delta differently.
    # The `generate_track_from_events` starts at 0.
    # So we can just append the note messages to track0.
    
    melody_events.sort(key=lambda x: x['tick'])
    last_tick = 0
    for e in melody_events:
        delta = max(0, e['tick'] - last_tick)
        track0.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta, channel=0))
        last_tick = e['tick']

    # 3. Rhythm Track
    track1 = mido.MidiTrack()
    mid.tracks.append(track1)
    track1.append(mido.MetaMessage('track_name', name='Drums', time=0))
    # Channel 9 (10th channel) is drums.
    
    rhythm_events = []
    onsets = rhythm_data.get('onsets', [])
    
    # Add steady hi-hat
    total_duration = times[-1] - start_time + 2.0
    beat_interval = 60 / bpm
    current_t = 0
    while current_t < total_duration:
        tick = int(current_t * ticks_per_second)
        rhythm_events.append({'tick': tick, 'type': 'note_on', 'note': 42, 'velocity': 50}) # Closed Hi-hat
        rhythm_events.append({'tick': tick + 50, 'type': 'note_off', 'note': 42, 'velocity': 0})
        current_t += beat_interval

    # Add kicks on peaks
    for t in onsets:
        tick = int((t - start_time) * ticks_per_second)
        # Kick (36) + Crash (49)
        rhythm_events.append({'tick': tick, 'type': 'note_on', 'note': 36, 'velocity': 110})
        rhythm_events.append({'tick': tick + 100, 'type': 'note_off', 'note': 36, 'velocity': 0})
        rhythm_events.append({'tick': tick, 'type': 'note_on', 'note': 49, 'velocity': 90}) 
        rhythm_events.append({'tick': tick + 100, 'type': 'note_off', 'note': 49, 'velocity': 0})

    # Generate Drum Track Content
    rhythm_events.sort(key=lambda x: x['tick'])
    last_tick = 0
    for e in rhythm_events:
        delta = max(0, e['tick'] - last_tick)
        track1.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta, channel=9))
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
