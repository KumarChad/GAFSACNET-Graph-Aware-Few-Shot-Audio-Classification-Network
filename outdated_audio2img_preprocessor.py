import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import librosa



# Paths
source_directory = r"D:\turing_playground\FSD50K.dev_audio"  # Original folder
target_directory = r"D:\turing_playground\FSD50K_subset"      # Folder to save valid files

# Create target directory if it doesn't exist
os.makedirs(target_directory, exist_ok=True)

# Function to check duration and copy audio files
def check_and_copy_audio(file_path, sr=22050, min_duration=1, max_duration=10):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr)
        duration = len(y) / sr  # Get actual duration
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False

    # Keep only files within the duration range
    if min_duration <= duration <= max_duration:
        # Copy file to the new folder
        dest_path = os.path.join(target_directory, os.path.basename(file_path))
        shutil.copy(file_path, dest_path)
        return True

    return False

# Get all .wav files
wav_files = [f for f in os.listdir(source_directory) if f.lower().endswith(".wav")]

# Process files
selected_count = 0

for wav_file in wav_files:
    file_path = os.path.join(source_directory, wav_file)
    
    if check_and_copy_audio(file_path):
        selected_count += 1
        print(f"Selected: {wav_file} (Total: {selected_count})")

print(f"\n Successfully saved {selected_count} files in {target_directory}")


input_file = r"D:\turing_playground\FSD50K_subset"
dest_file = r"D:\turing_playground\FSD50K_10k_data"
os.makedirs(dest_file, exist_ok=True)

# its Clear all files in the destination directory so that if there is any file before it gets deleted instead of adding dupicate values when run multiple times
for f in os.listdir(dest_file):
    file_path = os.path.join(dest_file, f)
    if os.path.isfile(file_path):
        os.remove(file_path)


wav_files = [f for f in os.listdir(input_file) if f.lower().endswith(".wav")]
random_files = random.sample(wav_files, 10000)

# it Copy's each randomly selected WAV file to the destination directory
for file in random_files:
    shutil.copy(os.path.join(input_file, file), os.path.join(dest_file, file))

# this one is to Verify the number of files copied
dest_files = [f for f in os.listdir(dest_file) if f.lower().endswith(".wav")]
if len(dest_files) == 10000:
    print("Success: 10000 files have been copied.")
else:
    print(f"Error: Expected 100 files, but found {len(dest_files)}.")
    
import soundfile as sf

# Directories
source_dir = r"D:\turing_playground\FSD50K_10k_data"
target_dir = r"D:\turing_playground\FSD50K_Padded"
os.makedirs(target_dir, exist_ok=True)

# Parameters
sr = 22050
max_duration = 10  # seconds
min_duration = 1  # seconds

# Process files
for file in os.listdir(source_dir):
    if not file.endswith(".wav"):
        continue

    file_path = os.path.join(source_dir, file)

    try:
        y, _ = librosa.load(file_path, sr=sr)
        duration = len(y) / sr
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

    # Skip files outside the duration range
    if duration < min_duration or duration > max_duration:
        print(f"Skipped {file} (out of range)")
        continue

    # Pad files shorter than 10 seconds
    if duration < max_duration:
        y = np.pad(y, (0, sr * max_duration - len(y)))

    # Save processed file
    sf.write(os.path.join(target_dir, file), y, sr)
    print(f"Processed and saved: {file}")

print("\nProcessing complete!")

from PIL import Image
import matplotlib.cm as cm

# Directories
input_audio_folder = r"D:\turing_playground\FSD50K_Padded"  # Input WAV files
#output_mfcc_folder = r"D:\turing_playground\FSD50K_mfcc"  # Output Spectrogram Images
output_resized_folder = r"D:\turing_playground\FSD50K_mfcc_resized"  # Folder for resized images

os.makedirs(output_resized_folder, exist_ok=True)

# Function to Generate & Save MFCC Spectrogram
def save_mfcc_spectrogram(y, sr, save_path, n_mfcc=40):
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    
    # Normalize
    mfcc_min = mfcc.min()
    mfcc_max = mfcc.max()
    normalized_mfcc = (mfcc - mfcc_min) / (mfcc_max - mfcc_min) * 255
    normalized_mfcc = normalized_mfcc.astype(np.uint8)
    normalized_mfcc_transposed = normalized_mfcc.T  # Transpose the MFCC data

    # 2. Replicate grayscale to RGB (Important!):
    cmap = cm.get_cmap('inferno')
    img_colored = cmap(normalized_mfcc_transposed / 255.0)[:, :, :3]  # Normalize before applying colormap
    img_colored = (img_colored * 255).astype(np.uint8)  # Convert to uint8

    # Convert to Image
    img = Image.fromarray(img_colored)  

    # Resize
    img = img.resize((512, 512), Image.Resampling.LANCZOS)

    # Save resized image
    img.save(save_path)

# Step 1: Process Audio Files & Save MFCC Spectrograms
for filename in os.listdir(input_audio_folder):
    if filename.lower().endswith(".wav"):
        audio_file_path = os.path.join(input_audio_folder, filename)
        print(f"Processing: {audio_file_path}")

        try:
            # Load Audio
            y, sr = librosa.load(audio_file_path, sr=22050)

            # Define Save Path
            spectrogram_filename = filename.replace(".wav", ".png")
            save_path = os.path.join(output_resized_folder, spectrogram_filename)

            # Save MFCC Spectrogram
            save_mfcc_spectrogram(y, sr, save_path)
            
        except Exception as e:
            print(f"Error processing {audio_file_path}: {e}")

print("\nMFCC Spectrogram Processing Completed!")
