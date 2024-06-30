import cv2
import numpy as np
import torch
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
import os
import librosa
from moviepy.editor import VideoFileClip, AudioFileClip
from scipy.signal import savgol_filter
from basicsr.archs.rrdbnet_arch import RRDBNet
import colorsys


def load_realesrgan_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = load_file_from_url(
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model_dir='weights',
        file_name='RealESRGAN_x4plus.pth',
        progress=True
    )
    upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=True)
    return upsampler

def enhance_frame(upsampler, img):
    torch.cuda.empty_cache()
    output, _ = upsampler.enhance(img, outscale=4)
    return output

# 0.0 o 1.0 = Rojo
# 0.083 = Naranja
# 0.167 = Amarillo
# 0.25 = Verde lima
# 0.333 = Verde
# 0.417 = Verde azulado
# 0.5 = Cian
# 0.583 = Azul claro
# 0.667 = Azul
# 0.75 = Púrpura
# 0.833 = Magenta
# 0.917 = Rosa


def resize_and_enhance_for_tiktok(input_file, audio_file, output_file, target_width=1080, target_height=1920, target_duration=10,hue=0.917):
    # Extraer audio del video de audio
    audio = VideoFileClip(audio_file).audio
    audio.write_audiofile("temp_audio.wav")

    print("Audio extracted, starting analysis...")
    y, sr = librosa.load("temp_audio.wav", duration=target_duration)
    print("Audio loaded, detecting beat...")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print("Beat detected, continuing...")
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Obtener la envolvente de energía del audio
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_env = savgol_filter(onset_env, 11, 3)  # Suavizar la envolvente

    print("Starting to load RealESRGAN model...")
    upsampler = load_realesrgan_model()
    print("RealESRGAN model loaded successfully.")

    print(f"Opening input video file: {input_file}")
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error al abrir el video")
        return
    print("Input video file opened successfully.")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: width={width}, height={height}, fps={fps}")
    aspect_ratio = target_width / target_height

    if width / height > aspect_ratio:
        new_width = int(height * aspect_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / aspect_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))

    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Starting to process video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or error reading frame.")
            break

        
        if width / height > aspect_ratio:
            new_width = int(height * aspect_ratio)
            new_height = height
            start_x = (width - new_width) // 2
            start_y = 0
        else:
            new_width = width
            new_height = int(width / aspect_ratio)
            start_x = 0
            start_y = (height - new_height) // 2

        # Crear una matriz de transformación para la traslación
        M = np.float32([[1, 0, -start_x], [0, 1, -start_y]])

        # Aplicar la traslación
        translated = cv2.warpAffine(frame, M, (new_width, new_height))

        resized = cv2.resize(translated, (target_width // 4, target_height // 4), interpolation=cv2.INTER_AREA)
        enhanced = enhance_frame(upsampler, resized)
        enhanced = cv2.resize(enhanced, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        frames.append(enhanced)
        frame_count += 1

        print(f"Procesando frame {frame_count}/{total_frames}", end='\r')

    print("\nProcesamiento de frames completado.")

    target_frames = int(target_duration * fps)
    output_frames = []
    current_beat = 0
    
    # Calcular la duración total del onset_env
    onset_duration = len(onset_env) * 512 / sr
    
    # Calcular cuántas veces necesitamos repetir el video
    repeat_count = int(np.ceil(target_frames / len(frames)))
    extended_frames = frames * repeat_count
    
    for i in range(target_frames):
        current_time = i / fps
        
        frame_index = int((current_time / target_duration) * len(frames)) % len(frames)
        
        beat_index = np.searchsorted(beat_times, current_time) % len(beat_times)
        time_since_last_beat = current_time - beat_times[beat_index - 1]
        time_to_next_beat = beat_times[beat_index] - current_time
        beat_progress = time_since_last_beat / (time_since_last_beat + time_to_next_beat)
        
        energy_index = int((current_time / onset_duration) * len(onset_env))
        energy_index = min(energy_index, len(onset_env) - 1)
        energy = onset_env[energy_index]
        
        frame = extended_frames[frame_index].copy()
        
        # Efecto de pulsación
        scale = 1 + 0.03 * np.sin(beat_progress * np.pi)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        h, w = frame.shape[:2]
        start_row = (h - target_height) // 2
        start_col = (w - target_width) // 2
        frame = frame[start_row:start_row+target_height, start_col:start_col+target_width]
        
        # Crear una capa de color basada en la energía
        color_intensity = np.clip(energy * 0.1, 0, 0.15) # Ajusta estos valores según necesites
        hsv_color = colorsys.hsv_to_rgb(hue, 1, 1)
        color_layer = np.full(frame.shape, [int(c * 255) for c in hsv_color[::-1]], dtype=np.uint8)
        
        # Crear una máscara de opacidad basada en la energía
        opacity = np.full(frame.shape[:2], int(color_intensity * 255), dtype=np.uint8)
        
        # Mezclar el frame original con la capa de color
        frame = cv2.addWeighted(frame, 1, color_layer, color_intensity, 0)
        
        output_frames.append(frame)
        
        if current_beat < len(beat_times) and current_time >= beat_times[current_beat]:
            current_beat += 1
    
    for frame in output_frames:
        out.write(frame)

    cap.release()
    out.release()

    print("Añadiendo audio al video...")
    # Añadir el audio al video procesado
    video = VideoFileClip(output_file)
    audio = AudioFileClip("temp_audio.wav").subclip(0, target_duration)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(f'{output_file[:-4]}_with_audio.mp4', codec="libx264", audio_codec="aac")

    # Limpiar archivos temporales
    os.remove("temp_audio.wav")
    os.remove(output_file)

    print("Proceso completado.")

# Uso del script
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
folder = 'Morenita'
input_file = folder+'/video.mp4'  # El video del que se tomarán las imágenes
audio_file = folder+'/audio.mp4'  # El video del que se tomará el audio
output_file = folder+'/tiktok_ready_video.mp4'
resize_and_enhance_for_tiktok(input_file, audio_file, output_file)