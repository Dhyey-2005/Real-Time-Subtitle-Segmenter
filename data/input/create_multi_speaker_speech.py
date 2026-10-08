"""Generate multi-speaker sample speech WAV with two distinct voices (Microsoft David and Microsoft Zira)."""
import os
import subprocess

out_path = r"F:\Real-Time Subtitle Segmenter\data\input\multi_speaker_test.wav"

ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{out_path}')

# Turn 1: Speaker 1 (David Desktop)
$synth.SelectVoice('Microsoft David Desktop')
$synth.Speak('Today we are discussing artificial intelligence and the renewable energy transition.')
Start-Sleep -Milliseconds 400

# Turn 2: Speaker 2 (Zira Desktop)
$synth.SelectVoice('Microsoft Zira Desktop')
$synth.Speak('Renewable energy transition is indeed a major topic for global climate policy.')
Start-Sleep -Milliseconds 400

# Turn 3: Speaker 1 (David Desktop)
$synth.SelectVoice('Microsoft David Desktop')
$synth.Speak('We must invest heavily into clean energy infrastructure and job creation initiatives.')
Start-Sleep -Milliseconds 400

# Turn 4: Speaker 2 (Zira Desktop)
$synth.SelectVoice('Microsoft Zira Desktop')
$synth.Speak('I completely agree that clean energy infrastructure drives long term economic growth.')

$synth.Dispose()
"""

subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True)
if os.path.exists(out_path):
    print(f"Generated multi-speaker speech WAV successfully: {out_path} ({os.path.getsize(out_path)} bytes)")
else:
    print("Failed to generate WAV.")
