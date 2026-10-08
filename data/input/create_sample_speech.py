"""Generate sample speech WAV using Windows SAPI."""
import os
import subprocess

out_path = r"F:\Real-Time Subtitle Segmenter\data\input\sample_speech.wav"
ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{out_path}')
$synth.Speak('Today we are discussing the new economic package and renewable energy transition. The government announced several new initiatives for job creation and public transit expansion. Renewable energy transition represents a major commitment.')
$synth.Dispose()
"""

subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True)
if os.path.exists(out_path):
    print(f"Generated sample speech WAV successfully: {out_path} ({os.path.getsize(out_path)} bytes)")
else:
    print("Failed to generate WAV.")
