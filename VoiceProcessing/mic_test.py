import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_plot(frame):
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)
    line.set_ydata(audio_data)
    
    return line,


# Parameters for the microphone test
FORMAT = pyaudio.paInt16  # Format for the audio
CHANNELS = 1  # Mono audio
RATE = 48000  # Sample rate (44.1kHz is standard for audio)
CHUNK = 12000  # Size of each audio chunk

p = pyaudio.PyAudio()

# Open the stream for recording
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(np.arange(0, CHUNK), np.arange(0, CHUNK))  # Initial plot
ax.set_title("Real-Time Audio Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")
ax.set_ylim(-2**15, 2**15)  # Range for 16-bit PCM audio data


# Create an animation to update the plot in real time
ani = animation.FuncAnimation(fig, update_plot, blit=True, interval=50)
plt.show()

# Clean up when done
stream.stop_stream()
stream.close()
p.terminate()