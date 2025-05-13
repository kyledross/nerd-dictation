import functools
import numpy as np
import pygame
import time

SAMPLE_RATE = 44100

def wait_for_completion(duration):
    """
    Wait for the specified duration in seconds.

    :param duration: Time to wait in seconds.
    """
    time.sleep(duration)


def apply_fade(audio_array, fade_duration):
    """
       Apply fade-in and fade-out effects to an audio array.

       :param audio_array: The numpy array representing the waveform.
       :param fade_duration: Duration of the fade-in and fade-out in seconds.
       :return: The audio array with fade-in and fade-out applied.
       """
    fade_samples = int(fade_duration * SAMPLE_RATE)
    # Create fade-in and fade-out envelopes
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    # Apply fade-in
    audio_array[:fade_samples] *= fade_in
    # Apply fade-out
    audio_array[-fade_samples:] *= fade_out

    return audio_array

@functools.lru_cache(maxsize=50)
def generate_sweep_sound(start_frequency, end_frequency, sound_duration):
    """
    Generates a smooth frequency sweep from start_frequency to end_frequency.

    :param start_frequency: The starting frequency in Hertz (Hz).
    :param end_frequency: The ending frequency in Hertz (Hz).
    :param sound_duration: The duration of the sweep in seconds.
    :return: A pygame Sound object of the sweep.
    """
    sound_duration = round(sound_duration * SAMPLE_RATE) / SAMPLE_RATE
    t = np.linspace(0, sound_duration, int(SAMPLE_RATE * sound_duration), endpoint=False)

    # Use a smooth interpolation for frequency (logarithmic chirp is smoother than linear)
    frequencies = np.logspace(np.log10(start_frequency), np.log10(end_frequency), num=len(t))

    # Create the waveform for the frequency sweep
    waveform = 0.5 * np.sin(2 * np.pi * frequencies * t)

    # Add 2nd and 3rd harmonics
    waveform += 0.5 * np.sin(2 * np.pi * 2 * frequencies * t)  # 2nd harmonic
    waveform += 0.25 * np.sin(2 * np.pi * 3 * frequencies * t)  # 3rd harmonic

    sound = create_sound_from_waveform(waveform)
    return sound


def create_sound_from_waveform(waveform, fade: bool = True) -> pygame.mixer.Sound:
    # add a fade for smoothness
    if fade:
        waveform = apply_fade(waveform, fade_duration=0.1)
    # Ensure starts and ends near zero
    if waveform[0] != 0:
        waveform[0] = 0
    if waveform[-1] != 0:
        waveform[-1] = 0
    # normalize to prevent clipping
    waveform /= np.max(np.abs(waveform))  # Normalize to range [-1, 1]
    # Convert the waveform to stereo
    stereo_wave = np.vstack((waveform, waveform)).T
    # Convert the waveform to a format suitable for pygame
    sound = pygame.sndarray.make_sound((32767 * stereo_wave).astype(np.int16).copy())
    return sound


def play_sweep(start_frequency, end_frequency, sound_duration, volume=0.1, wait=False):
    """
    Plays a frequency sweep from start_frequency to end_frequency.

    :param start_frequency: The starting frequency in Hertz (Hz).
    :param end_frequency: The ending frequency in Hertz (Hz).
    :param sound_duration: The duration of the sweep in seconds.
    :param volume: The volume level of the sound (0.0 to 1.0).
    :param wait: If True, wait for the sound to finish playing before continuing.
    :return: None
    """
    sound = generate_sweep_sound(start_frequency, end_frequency, sound_duration)
    sound.set_volume(volume)
    channel = pygame.mixer.find_channel()  # Get an available channel
    if channel:
        channel.play(sound)
        if wait:
            wait_for_completion(sound_duration)


@functools.lru_cache(maxsize=50)
def generate_rumble_sound(frequency, sound_duration, noise_intensity=0, noise_frequency=4):
    """
    Generate a low-frequency rumble sound with controllable noise frequency.
    :param frequency: The base frequency of the rumble in Hertz (Hz).
    :param sound_duration: The duration of the sound in seconds.
    :param noise_intensity: How much random noise is added to the sound (0.0 for pure tone, 1.0 for heavy noise).
    :param noise_frequency: The frequency of the noise variations in Hertz (Hz).
    :return: A pygame Sound object of the rumble.
    """
    t = np.linspace(0, sound_duration, int(SAMPLE_RATE * sound_duration), endpoint=False)

    # Base low-frequency wave (a slow sine wave)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Generate low-frequency "noise" using a sine wave combined with randomness
    low_frequency_noise = np.sin(2 * np.pi * noise_frequency * t)
    random_amplitude = np.random.random(size=t.shape) - 0.5  # Random variations
    noise = noise_intensity * low_frequency_noise * random_amplitude

    # Combine the wave and the noise
    rumble = wave + noise

    sound = create_sound_from_waveform(rumble, fade=False)
    return sound


def play_rumble(sound_duration, frequency=60, noise_intensity=0, volume=0.1, wait=False):
    """
    Play a low-frequency rumble sound.

    :param frequency: The base frequency of the rumble in Hertz (Hz).
    :param sound_duration: The duration of the rumble in seconds.
    :param noise_intensity: How much random noise to add for a "rumble effect."
    :param volume: The volume level of the rumble sound (0.0 to 1.0).
    :param wait: If True, wait for the sound to finish playing before continuing.
    :return: None
    """
    sound = generate_rumble_sound(frequency, sound_duration, noise_intensity)
    sound.set_volume(volume)
    sound.play()
    if wait:
        wait_for_completion(sound_duration)


@functools.lru_cache(maxsize=50)
def generate_plop_sound(start_frequency, end_frequency, sound_duration=0.2):
    """
    Generate a 'plop' sound effect that starts at a low frequency and sweeps to a higher frequency.

    :param start_frequency: The starting frequency in Hertz (Hz).
    :param end_frequency: The ending frequency in Hertz (Hz).
    :param sound_duration: The total duration of the plop sound, in seconds.
    :return: A pygame Sound object of the plop.
    """
    t = np.linspace(0, sound_duration, int(SAMPLE_RATE * sound_duration), endpoint=False)

    # Generate a logarithmic sweep for rising pitch effect (sounds smooth to the ear)
    frequencies = np.logspace(np.log10(start_frequency), np.log10(end_frequency), len(t))
    wave = 0.5 * np.sin(2 * np.pi * frequencies * t)

    sound = create_sound_from_waveform(wave)
    return sound


def play_plop(start_frequency=150.0, end_frequency=400.0, sound_duration=0.15, volume=0.2, wait=False):
    """
    Play a short 'plop' sound effect.

    :param start_frequency: The starting frequency in Hertz (Hz).
    :param end_frequency: The ending frequency in Hertz (Hz).
    :param sound_duration: The duration of the plop sound, in seconds.
    :param volume: The volume level of the plop sound (0.0 to 1.0).
    :param wait: If True, wait for the sound to finish playing before continuing.
    :return: None
    """
    sound = generate_plop_sound(start_frequency, end_frequency, sound_duration)
    sound.set_volume(volume)
    sound.play()
    if wait:
        wait_for_completion(sound_duration)


@functools.lru_cache(maxsize=50)
def generate_fanfare_sound(start_frequencies, note_duration=0.2):
    """
    Generate a celebratory "ta-da!" fanfare sound using a sequence of rising notes with harmonies.

    :param start_frequencies: A list or tuple of frequencies (in Hz) for each note in the fanfare.
    :param note_duration: The duration of each note in the fanfare, in seconds.
    :return: A pygame Sound object of the fanfare.
    """
    if isinstance(start_frequencies, list):
        start_frequencies = tuple(start_frequencies)

    # Harmonies to include for each root note frequency (major third, perfect fifth)
    harmony_intervals = [1.25, 1.5]  # Multipliers for the root frequency

    total_duration = len(start_frequencies) * note_duration
    t = np.linspace(0, total_duration, int(SAMPLE_RATE * total_duration), endpoint=False)

    # Initialize an empty waveform
    waveform = np.zeros_like(t)

    # Overlay each note with harmonics into the waveform
    for i, frequency in enumerate(start_frequencies):
        # Time indices for this note
        start_idx = int(i * note_duration * SAMPLE_RATE)
        end_idx = int((i + 1) * note_duration * SAMPLE_RATE)
        note_t = t[:end_idx - start_idx]  # Time range for this note

        # Root frequency for the note
        root_wave = 0.5 * np.sin(2 * np.pi * frequency * note_t)

        # Add harmonic frequencies (major third and perfect fifth)
        harmonics_wave = sum(
            0.25 * np.sin(2 * np.pi * (frequency * interval) * note_t) for interval in harmony_intervals
        )

        # Combine the root note with its harmonics
        note_wave = root_wave + harmonics_wave

        # Add the note (with harmonics) to the full waveform
        waveform[start_idx:end_idx] += note_wave

    sound = create_sound_from_waveform(waveform)

    return sound


def play_start_sound(start_frequencies=None, note_duration=0.15, volume=0.3, wait=False):
    """
    Play a celebratory "ta-da!" fanfare sound.

    :param start_frequencies: A list of frequencies (in Hz) for each note in the fanfare.
    :param note_duration: The duration of each note, in seconds.
    :param volume: The volume level for the fanfare.
    :param wait: If True, wait for the sound to finish playing before continuing.
    :return: None
    """
    if start_frequencies is None:
        start_frequencies = [400.0, 600.0]

    if isinstance(start_frequencies, list):
        start_frequencies = tuple(start_frequencies)

    sound = generate_fanfare_sound(start_frequencies, note_duration)
    sound.set_volume(volume)
    sound.play()
    if wait:
        total_duration = len(start_frequencies) * note_duration
        wait_for_completion(total_duration)


def play_end_sound(start_frequencies=None, note_duration=0.15, volume=0.3, wait=False):
    """
    Play a celebratory "ta-da!" fanfare sound.

    :param start_frequencies: A list of frequencies (in Hz) for each note in the fanfare.
    :param note_duration: The duration of each note, in seconds.
    :param volume: The volume level for the fanfare.
    :param wait: If True, wait for the sound to finish playing before continuing.
    :return: None
    """
    if start_frequencies is None:
        start_frequencies = [600.0, 400.0]

    if isinstance(start_frequencies, list):
        start_frequencies = tuple(start_frequencies)

    sound = generate_fanfare_sound(start_frequencies, note_duration)
    sound.set_volume(volume)
    sound.play()
    if wait:
        total_duration = len(start_frequencies) * note_duration
        wait_for_completion(total_duration)


@functools.lru_cache(maxsize=50)
def generate_bonk_sound():
    """
    Generate a 'bonk' sound effect for indicating a wrong move, with added distortion.
    :return: A pygame Sound object of the bonk.
    """
    base_frequency=70.0
    sound_duration=0.2
    decay=0.02
    volume = 0.6
    t = np.linspace(0, sound_duration, int(SAMPLE_RATE * sound_duration), endpoint=False)

    # Generate a decaying sine wave for the bonk sound
    wave = 0.5 * np.sin(2 * np.pi * base_frequency * t) * np.exp(-decay * t)

    # Add distortion to the waveform (simulate "not okay" sound)
    wave = np.tanh(5 * wave)

    sound = create_sound_from_waveform(wave)
    sound.set_volume(volume)
    return sound


def play_bonk(wait=False):
    """
    Play a short 'bonk' sound effect.

    :param wait: If True, wait for the sound to finish playing before continuing.
    :return: None
    """
    sound = generate_bonk_sound()
    sound.play()
    if wait:
        wait_for_completion(0.2)


@functools.lru_cache(maxsize=50)
def generate_lost_game_sound(high_frequency, low_frequency, high_note_duration=0.2, low_note_duration=0.5):
    """
    Generate a sorrowful two-note harmonic sound that signifies loss.

    :param high_frequency: The frequency (in Hz) of the first high note.
    :param low_frequency: The frequency (in Hz) of the second low note.
    :param high_note_duration: The duration of the first high note, in seconds.
    :param low_note_duration: The duration of the second low note, in seconds.
    :return: A pygame Sound object of the sorrowful sound.
    """
    # Harmonies for a sorrowful effect (minor third and minor seventh intervals)
    harmony_intervals = [1.2, 0.5]  # Harmonic intervals

    total_duration = high_note_duration + low_note_duration
    t = np.linspace(0, total_duration, int(SAMPLE_RATE * total_duration), endpoint=False)

    # Initialize an empty waveform
    waveform = np.zeros_like(t)

    # Add the high note and its harmonies
    high_start_idx = 0
    high_end_idx = int(high_note_duration * SAMPLE_RATE)

    # noinspection DuplicatedCode
    high_t = t[high_start_idx:high_end_idx]
    high_root_wave = 0.5 * np.sin(2 * np.pi * high_frequency * high_t)
    high_harmonics_wave = sum(
        0.25 * np.sin(2 * np.pi * (high_frequency * interval) * high_t) for interval in harmony_intervals
    )
    waveform[high_start_idx:high_end_idx] += high_root_wave + high_harmonics_wave

    # Add the low note and its harmonies
    low_start_idx = high_end_idx
    low_end_idx = low_start_idx + int(low_note_duration * SAMPLE_RATE)

    # Correctly generate `low_t` based on the waveform's slice indices
    # noinspection DuplicatedCode
    low_t = t[low_start_idx:low_end_idx]

    low_root_wave = 0.5 * np.sin(2 * np.pi * low_frequency * low_t)
    low_harmonics_wave = sum(
        0.25 * np.sin(2 * np.pi * (low_frequency * interval) * low_t) for interval in harmony_intervals
    )
    waveform[low_start_idx:low_end_idx] += low_root_wave + low_harmonics_wave

    sound = create_sound_from_waveform(waveform)
    return sound


def play_lost_game_sound(high_frequency=600.0, low_frequency=200.0, high_note_duration=0.2, low_note_duration=0.5,
                         volume=0.3, wait=False):
    """
    Play a sorrowful two-note harmonic sound that signifies loss.

    :param high_frequency: The frequency (in Hz) of the first high note.
    :param low_frequency: The frequency (in Hz) of the second low note.
    :param high_note_duration: The duration of the first high note, in seconds.
    :param low_note_duration: The duration of the second low note, in seconds.
    :param volume: The volume level for the sound (0.0 to 1.0).
    :param wait: If True, wait for the sound to finish playing before continuing.
    :return: None
    """
    sound = generate_lost_game_sound(high_frequency, low_frequency, high_note_duration, low_note_duration)
    sound.set_volume(volume)
    sound.play()
    if wait:
        wait_for_completion(high_note_duration + low_note_duration)
