import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import ttkbootstrap as ttk
import os
import threading
import subprocess
import tempfile
from typing import Optional, List, Dict, Tuple
import time
import nltk
import torch
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import google.generativeai as genai
import shutil
import errno
from gtts import gTTS
import psutil


class LinguoAIVideoVoicePro:
    def __init__(self):
        # --- Parámetros de configuración ---
        self.ffmpeg_timeout = 30
        self.process_timeout = 7200
        self.chunk_timeout = 600
        self.validate_chunk_timeout = 20
        self.audio_extraction_chunk_size = 60
        self.transcribe_chunk_size = 60
        self.tts_batch_size = 10
        self.model_size = "medium"
        self.target_language = "en" # Idioma objetivo predeterminado es inglés
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gemini_api_key = ""
        self.tts_engine = "gtts"
        self.hw_accel_info = self.detect_hardware_acceleration()  # Guardar la información de aceleración de hardware
        self.hw_accel = self.hw_accel_info['accel'] if self.hw_accel_info else None

        # --- Estado interno ---
        self.whisper_model: Optional[WhisperModel] = None
        self.current_process: Optional[subprocess.Popen] = None
        self.process_start_time: Optional[float] = None
        self.process_temp_dir: str = ""

        # --- Componentes de la GUI ---
        self.root = ttk.Window(themename="darkly")
        self.root.title("LinguoAI VideoVoice Pro")
        self.root.geometry("640x850")  # Establecer tamaño inicial

        self.source_file = tk.StringVar()
        self.target_file = tk.StringVar()
        self.selected_language = tk.StringVar(value="en") # Idioma seleccionado predeterminado es inglés
        self.gemini_key = tk.StringVar()  # Clave API de Gemini
        self.progress_var = tk.DoubleVar(value=0)
        self.log_messages: List[str] = []

        self.languages = {
            "af": ("Afrikaans", "🇿🇦"),
            "sq": ("Albanés", "🇦🇱"),
            "am": ("Amhárico", "🇪🇹"),
            "ar": ("Árabe", "🇸🇦"),
            "eu": ("Vasco", "🇪🇸"),
            "bn": ("Bengalí", "🇧🇩"),
            "bs": ("Bosnio", "🇧🇦"),
            "bg": ("Búlgaro", "🇧🇬"),
            "ca": ("Catalán", "🇦🇩"),
            "zh-CN": ("Chino (Simplificado)", "🇨🇳"),
            "zh-TW": ("Chino (Tradicional)", "🇹🇼"),
            "hr": ("Croata", "🇭🇷"),
            "cs": ("Checo", "🇨🇿"),
            "da": ("Danés", "🇩🇰"),
            "nl": ("Neerlandés", "🇳🇱"),
            "en": ("Inglés", "🇬🇧"),
            "et": ("Estonio", "🇪🇪"),
            "tl": ("Filipino", "🇵🇭"),
            "fi": ("Finlandés", "🇫🇮"),
            "fr": ("Francés", "🇫🇷"),
            "gl": ("Gallego", "🇪🇸"),
            "de": ("Alemán", "🇩🇪"),
            "el": ("Griego", "🇬🇷"),
            "gu": ("Gujarati", "🇮🇳"),
            "ha": ("Hausa", "🇳🇬"),
            "he": ("Hebreo", "🇮🇱"),
            "hi": ("Hindi", "🇮🇳"),
            "hu": ("Húngaro", "🇭🇺"),
            "is": ("Islandés", "🇮🇸"),
            "id": ("Indonesio", "🇮🇩"),
            "it": ("Italiano", "🇮🇹"),
            "ja": ("Japonés", "🇯🇵"),
            "jw": ("Javanés", "🇮🇩"),
            "kn": ("Kannada", "🇮🇳"),
            "km": ("Jemer", "🇰🇭"),
            "ko": ("Coreano", "🇰🇷"),
            "la": ("Latín", "🇻🇦"),
            "lv": ("Letón", "🇱🇻"),
            "lt": ("Lituano", "🇱🇹"),
            "ms": ("Malayo", "🇲🇾"),
            "mr": ("Marathi", "🇮🇳"),
            "ml": ("Malayalam", "🇮🇳"),
            "my": ("Myanmar (Birmano)", "🇲🇲"),
            "ne": ("Nepalí", "🇳🇵"),
            "no": ("Noruego", "🇳🇴"),
            "pa": ("Punjabi", "🇮🇳"),
            "pl": ("Polaco", "🇵🇱"),
            "pt": ("Portugués", "🇵🇹"),
            "ro": ("Rumano", "🇷🇴"),
            "ru": ("Ruso", "🇷🇺"),
            "sr": ("Serbio", "🇷🇸"),
            "si": ("Cingalés", "🇱🇰"),
            "sk": ("Eslovaco", "🇸🇰"),
            "sl": ("Esloveno", "🇸🇮"),
            "es": ("Español", "🇪🇸"),
            "su": ("Sundanés", "🇮🇩"),
            "sw": ("Suajili", "🇰🇪"),
            "sv": ("Sueco", "🇸🇪"),
            "ta": ("Tamil", "🇮🇳"),
            "te": ("Telugu", "🇮🇳"),
            "th": ("Tailandés", "🇹🇭"),
            "tr": ("Turco", "🇹🇷"),
            "uk": ("Ucraniano", "🇺🇦"),
            "ur": ("Urdu", "🇵🇰"),
            "vi": ("Vietnamita", "🇻🇳"),
            "cy": ("Galés", "🇬🇧")
        }

        # --- Inicialización ---
        self.setup_gui()
        self.setup_ffmpeg()
        self.load_api_keys_from_environment()
        self.load_whisper_model()
        if self.gemini_api_key:
            self.init_gemini()
        self.log_hardware_acceleration()

    def seconds_to_srt_time(self, seconds: float) -> str:
        """Convierte segundos al formato de tiempo SRT (HH:MM:SS,mmm)."""
        milliseconds = int((seconds * 1000) % 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    def create_srt_file(self, segments: List[Dict], output_path: str):
        """Crea un archivo SRT a partir de los segmentos transcritos/traducidos."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments):
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text']

                    # Convierte segundos al formato de tiempo SRT
                    start_time_srt = self.seconds_to_srt_time(start_time)
                    end_time_srt = self.seconds_to_srt_time(end_time)

                    f.write(f"{i + 1}\n")
                    f.write(f"{start_time_srt} --> {end_time_srt}\n")
                    f.write(f"{text}\n\n")

            self.log_message(f"Archivo SRT creado: {output_path}")

        except Exception as e:
            self.log_message(f"Error al crear el archivo SRT: {e}")

    def log_hardware_acceleration(self):
        """Registra la información de aceleración de hardware."""
        if self.hw_accel_info:
            self.log_message(f"Aceleración por hardware detectada: {self.hw_accel_info['accel']} ({self.hw_accel_info['info']})")
        else:
            self.log_message("No se detectó aceleración por hardware.")

    def detect_hardware_acceleration(self):
        """Detecta la aceleración por hardware (NVIDIA, Intel, AMD)."""
        try:
            # NVIDIA
            try:
                subprocess.run(['nvidia-smi'], check=True, capture_output=True)
                return {'accel': 'cuda', 'info': 'GPU NVIDIA detectada'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # Intel (Quick Sync)
            try:
                output = subprocess.run(['vainfo'], check=True, capture_output=True, text=True).stdout
                if "VA-API" in output:  # Verificación muy general, se puede refinar
                    return {'accel': 'qsv', 'info': 'Intel Quick Sync detectado'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # AMD (AMF) - un poco más complejo, ya que no hay un "amfinfo" simple
            # Se podría buscar controladores/dispositivos específicos, pero eso es específico del SO.
            # Aquí una verificación muy simple e incompleta para Linux:
            if os.name == 'posix':  # Linux/macOS
                try:
                    output = subprocess.run(['lspci', '-v'], check=True, capture_output=True, text=True).stdout
                    if "Advanced Micro Devices, Inc. [AMD/ATI]" in output:
                        return {'accel': 'h264_vaapi', 'info': 'GPU AMD detectada (VAAPI)'}  # ¡Suposición!
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass

            return None  # No se encontró aceleración por hardware

        except Exception as e:
            self.log_message(f"Error al detectar la aceleración por hardware: {e}")
            return None
    def init_gemini(self):
        """Inicializa el modelo Gemini Pro."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.log_message("Modelo Gemini Pro inicializado.")
        except Exception as e:
            self.log_message(f"Error al inicializar Gemini Pro: {e}")
            messagebox.showerror("Error de Gemini", f"No se pudo inicializar Gemini Pro: {e}")
            self.model = None  # Establecer a None en caso de fallo

    def check_process_timeout(self):
        """Verifica si el proceso general ha excedido el tiempo máximo permitido."""
        if self.process_start_time and time.time() - self.process_start_time > self.process_timeout:
            if self.current_process:
                self.log_message(f"¡Tiempo de espera del proceso agotado! Matando proceso (PID: {self.current_process.pid})...")
                self.kill_process_tree(self.current_process.pid)  # ¡Matar árbol de procesos!
            raise TimeoutError("El proceso excedió el tiempo máximo permitido")

    def kill_process_tree(self, pid):
        """Mata un proceso y todos sus procesos hijo."""
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):  # Obtener todos los hijos/nietos
                self.log_message(f"Matando proceso hijo: {child.pid}")
                child.kill()
            self.log_message(f"Matando proceso padre: {parent.pid}")
            parent.kill()
        except psutil.NoSuchProcess:
            self.log_message(f"No se encontró el proceso con PID {pid}.")
        except Exception as e:
            self.log_message(f"Error al matar el árbol de procesos: {e}")

    def run_subprocess_with_timeout(self, command, timeout, error_message):
        """Ejecuta un subproceso con tiempo de espera dinámico y verificaciones de actividad."""
        try:
            self.log_message(f"Ejecutando comando con tiempo de espera {timeout}: {' '.join(command)}")
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.log_message(f"Proceso iniciado con PID: {self.current_process.pid}")

            start_time = time.time()
            last_output_time = start_time

            stdout, stderr = self.current_process.communicate(timeout=timeout)  # ¡Usar communicate!
            retcode = self.current_process.returncode

            if retcode != 0:
                self.log_message(f"El proceso falló con código de error {retcode}:")
                self.log_message(f"Stdout: {stdout}")
                self.log_message(f"Stderr: {stderr}")
                raise subprocess.CalledProcessError(retcode, command, stdout, stderr)

            self.log_message("Comando completado con éxito.")
            return stdout, stderr

        except subprocess.TimeoutExpired:
            self.log_message(f"Tiempo de espera del proceso agotado después de {timeout} segundos")
            self.kill_process_tree(self.current_process.pid)  # ¡Matar árbol de procesos!
            stdout, stderr = self.current_process.communicate()  # Obtener cualquier salida restante
            self.log_message(f"Stdout: {stdout}")
            self.log_message(f"Stderr: {stderr}")
            raise TimeoutError(error_message)

        except Exception as e:
            self.log_message(f"Ocurrió un error inesperado: {e}")
            if self.current_process:
                self.kill_process_tree(self.current_process.pid)  # Matar si todavía se está ejecutando
            raise
        finally:
            self.current_process = None

    def extract_audio_chunk(self, video_path, audio_path, start_time, duration):
        """Extrae un fragmento de audio del video."""
        command = [
            "ffmpeg",
            "-y",  # Sobrescribir archivos de salida sin preguntar
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-vn",  # Deshabilitar video
            "-acodec", "libmp3lame",
            "-q:a", "2",  # MP3 de buena calidad
            "-loglevel", "error",  # Reducir la verbosidad
            audio_path
        ]
        if self.hw_accel:
            command.insert(1, "-hwaccel")
            command.insert(2, self.hw_accel)

        try:
            self.run_subprocess_with_timeout(command, self.chunk_timeout, "Tiempo de espera agotado para la extracción de audio")
            self.log_message(f"Fragmento de audio extraído: '{audio_path}'")
        except Exception as e:
            self.log_message(f"Error al extraer el fragmento de audio: {e}")
            raise

    def batch_segments(self, segments: List[Dict]) -> List[List[Dict]]:
        """Divide los segmentos en lotes más pequeños para TTS."""
        batched_segments = []
        for i in range(0, len(segments), self.tts_batch_size):
            batch = segments[i:i + self.tts_batch_size]
            batched_segments.append(batch)
        return batched_segments

    def validate_audio_chunk(self, chunk_path: str) -> bool:
        """Valida un fragmento de audio utilizando ffprobe."""
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            chunk_path
        ]
        try:
            stdout, stderr = self.run_subprocess_with_timeout(
                command,
                self.validate_chunk_timeout,
                f"Tiempo de espera agotado para la validación del fragmento de audio: {chunk_path}"
            )
            if stdout.strip():
                self.log_message(f"Fragmento de audio validado: {chunk_path}")
                return True
            else:
                self.log_message(f"Validación del fragmento de audio fallida (sin duración): {chunk_path}")
                return False
        except Exception as e:
            self.log_message(f"Error al validar el fragmento de audio {chunk_path}: {e}")
            return False

    def merge_audio_chunks(self, audio_chunks, output_path):
        """Fusiona varios fragmentos de audio en un solo archivo utilizando FFmpeg."""
        if not audio_chunks:
            self.log_message("No hay fragmentos de audio para fusionar.")
            return

        valid_chunks = [chunk for chunk in audio_chunks if self.validate_audio_chunk(chunk)]
        if not valid_chunks:
            self.log_message("No hay fragmentos de audio válidos para fusionar.")
            return

        timestamp = int(time.time())
        temp_dir = os.path.join(tempfile.gettempdir(), f"audio_merge_temp_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)

        list_file_path = os.path.join(temp_dir, "chunk_list.txt")
        try:
            with open(list_file_path, "w") as f:
                for chunk_path in valid_chunks:
                    abs_chunk_path = os.path.abspath(chunk_path)  # Usar ruta absoluta
                    f.write(f"file '{abs_chunk_path}'\n")

            command = [
                "ffmpeg",
                "-y",  # Sobrescribir
                "-f", "concat",
                "-safe", "0",  # Requerido para rutas absolutas con concat
                "-i", list_file_path,
                "-c", "copy",
                "-loglevel", "error",
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            # Calcular un tiempo de espera dinámico basado en el número de fragmentos.
            merge_timeout = len(valid_chunks) * 10 + 30  # 10 segundos por fragmento + 30 base
            self.run_subprocess_with_timeout(command, merge_timeout, "Tiempo de espera agotado para la fusión de audio")
            self.log_message(f"Fragmentos de audio fusionados: '{output_path}'")

        except Exception as e:
            self.log_message(f"Error al fusionar fragmentos de audio: {e}")
            raise
        finally:
            self.remove_file_with_retry(list_file_path)  # Usar la función de reintento
            self.remove_directory_with_retry(temp_dir)  # y para el directorio

    def merge_video_audio(self, audio_file):
        """Fusiona el audio final con el video original."""
        try:
            output_path = self.target_file.get()
            video_path = self.source_file.get()

            # Obtener la duración del video utilizando ffprobe
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Error al obtener la duración del video")
            total_duration = float(duration_output.strip())
            self.log_message(f"Duración del video para la fusión: {total_duration:.2f} segundos")

            # Calcular tiempo de espera dinámico (ej., 3x duración + 120 segundos)
            dynamic_timeout = int(3 * total_duration + 120)
            self.log_message(f"Tiempo de espera dinámico para la fusión: {dynamic_timeout} segundos")

            command = [
                'ffmpeg',
                '-y',  # Sobrescribir archivo de salida
                '-i', video_path,
                '-i', audio_file,
                '-c:v', 'h264_nvenc' if self.hw_accel == 'cuda' else 'libx264',  # H.265
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',  # Terminar la codificación cuando la transmisión más corta termine
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            self.run_subprocess_with_timeout(command, dynamic_timeout, "Tiempo de espera agotado para la fusión de video y audio")
            self.log_message(f"Video y audio fusionados: {output_path}")
        except Exception as e:
            self.log_message(f"Error al fusionar video y audio: {e}")
            raise

    def shorten_text_with_gemini(self, text: str) -> str:
        """Acorta el texto utilizando Gemini Pro, manejando errores y límites de velocidad."""
        if self.model is None:
            self.log_message("Modelo Gemini Pro no inicializado. Omitiendo el acortamiento.")
            return text

        try:
            prompt = f"Por favor, acorta el siguiente texto manteniendo la información clave:\n\n{text}"
            time.sleep(1.5)  # Límite de velocidad: Pausa de 1.5 segundos
            response = self.model.generate_content(prompt)
            if response and response.text:
                shortened_text = response.text
                self.log_message("Texto acortado con Gemini Pro.")
                return shortened_text
            else:
                self.log_message("Gemini Pro devolvió una respuesta vacía. Usando el texto original.")
                return text
        except Exception as e:
            self.log_message(f"Error al acortar el texto con Gemini: {e}")
            return text

    def process_video(self):
        """Flujo de trabajo principal de procesamiento de video."""
        self.process_start_time = time.time()
        timestamp = int(time.time())
        self.process_temp_dir = os.path.join(tempfile.gettempdir(), f"process_temp_{timestamp}")
        os.makedirs(self.process_temp_dir, exist_ok=True)

        try:
            self.log_message("Iniciando procesamiento de video...")
            self.progress_var.set(5)

            # Configurar la verificación de tiempo de espera *antes* de iniciar cualquier tarea.
            def check_timeout():
                self.check_process_timeout()
                self.root.after(1000, check_timeout)  # Verificar cada segundo

            check_timeout()  # Iniciar el verificador de tiempo de espera

            self.progress_var.set(10)
            audio_chunks, total_duration = self.extract_audio_in_chunks(self.source_file.get())
            self.progress_var.set(25)

            segments, total_words_original = self.transcribe_audio_in_chunks(audio_chunks, self.target_language)
            if not segments:
                raise Exception("La transcripción falló.")
            self.progress_var.set(45)

            translated_segments, total_words_translated = self.translate_and_refine_segments(segments)

            # --- Acortamiento de texto con Gemini (Opcional) ---
            total_words_shortened = 0
            if self.gemini_api_key and self.model:
                shortened_segments = []
                for segment in translated_segments:
                    shortened_text = self.shorten_text_with_gemini(segment['text'])
                    shortened_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': shortened_text
                    })
                    total_words_shortened += len(shortened_text.split())
                    self.root.update()  # Actualizar GUI
                translated_segments = shortened_segments

                shortening_change = (
                    ((total_words_translated - total_words_shortened) / total_words_translated) * 100
                    if total_words_translated else 0
                )
                self.log_message(f"El acortamiento redujo el conteo de palabras en: {shortening_change:.2f}%")
            else:
                total_words_shortened = total_words_translated
                self.log_message("Clave API de Gemini no proporcionada. Omitiendo el acortamiento.")

            self.progress_var.set(60)

            # Crear el archivo SRT (Ejemplo)
            srt_output_path = os.path.splitext(self.target_file.get())[0] + ".srt"  # Coincide con el nombre del video de salida
            self.create_srt_file(translated_segments, srt_output_path)

            # --- Informe de conteo de palabras ---
            if total_words_original > 0:
                translation_change = (total_words_translated / total_words_original) * 100
                self.log_message(f"Conteo de palabras original: {total_words_original}")
                self.log_message(f"Conteo de palabras traducido: {total_words_translated}")
                self.log_message(f"Cambio en el conteo de palabras de la traducción: {translation_change:.2f}%")
            else:
                self.log_message("El conteo de palabras original es cero. Omitiendo porcentaje.")

            # --- TTS y fusión de audio ---
            batched_translated_segments = self.batch_segments(translated_segments)
            all_updated_segments = []
            merged_audio_files = []

            for i, batch in enumerate(batched_translated_segments):
                self.log_message(f"Procesando lote TTS {i+1}/{len(batched_translated_segments)}")
                merged_audio_path, updated_segments = self.generate_tts_audio_for_segments(batch)
                if merged_audio_path:
                    all_updated_segments.extend(updated_segments)
                    merged_audio_files.append(merged_audio_path)
                else:
                    self.log_message(f"Lote TTS {i+1} falló.")
                    # Limpiar *todos* los archivos TTS creados previamente en caso de fallo
                    for file_path in merged_audio_files:
                        self.remove_file_with_retry(file_path)
                    raise Exception(f"La generación de audio TTS falló para el lote {i + 1}.")

            # --- Fusión final de audio (si hay varios lotes) ---
            if len(merged_audio_files) > 1:
                final_merged_audio_path = os.path.join(self.process_temp_dir,
                                                        f"final_merged_tts_audio_{timestamp}.mp3")
                self.merge_audio_chunks(merged_audio_files, final_merged_audio_path)
                self.log_message(f"Audio TTS fusionado final: {final_merged_audio_path}")
            elif merged_audio_files:
                final_merged_audio_path = merged_audio_files[0]
                self.log_message(f"Audio TTS fusionado final (lote único): {final_merged_audio_path}")
            else:
                raise Exception("No se generó audio TTS.")

            # --- Limpiar archivos TTS intermedios ---
            for file_path in merged_audio_files:
                if file_path != final_merged_audio_path:  # ¡No borrar el archivo final!
                    self.remove_file_with_retry(file_path)

            self.progress_var.set(80)

            # --- Fusión final de video/audio ---
            self.merge_video_audio(final_merged_audio_path)
            self.progress_var.set(95)

            self.log_message("¡Procesamiento completado con éxito! 🎉")
            messagebox.showinfo("Éxito", "¡Procesamiento de video completado!")


        except TimeoutError as e:
            self.log_message(f"Tiempo de espera de procesamiento agotado: {str(e)}")
            messagebox.showerror("Error", f"Tiempo de espera de procesamiento agotado: {str(e)}")
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"El procesamiento falló: {str(e)}")
        finally:
            # --- Limpieza ---
            self.progress_var.set(0)
            self.process_start_time = None
            self.current_process = None
            self.start_button.config(state=tk.NORMAL)  # Volver a habilitar el botón
            self.remove_directory_with_retry(self.process_temp_dir)

    def remove_file_with_retry(self, file_path, retries=3, delay=0.5):
        """Elimina un archivo, reintentando si es necesario."""
        file_path = os.path.abspath(file_path)  # Usar ruta absoluta
        for i in range(retries):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                self.log_message(f"Archivo eliminado: {file_path}")
                return  # Éxito
            except OSError as e:
                if e.errno == errno.ENOENT:  # Archivo no encontrado - ya eliminado
                    self.log_message(f"Archivo no encontrado (ya eliminado): {file_path}")
                    return
                if i < retries - 1:  # No esperar en el último intento
                    self.log_message(f"Reintentando la eliminación del archivo ({i+1}/{retries}): {file_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Error al eliminar el archivo después de varios reintentos: {file_path} - {e}")
            except Exception as e:
                self.log_message(f"Error inesperado al eliminar el archivo: {file_path} - {e}")
                return  # No reintentar para errores inesperados

    def remove_directory_with_retry(self, dir_path, retries=5, delay=1):
        """Elimina un directorio, reintentando si es necesario (especialmente para no vacíos)."""
        dir_path = os.path.abspath(dir_path)
        for i in range(retries):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                self.log_message(f"Directorio eliminado: {dir_path}")
                return
            except OSError as e:
                if e.errno == errno.ENOENT:  # Directorio ya eliminado
                    self.log_message(f"Directorio ya eliminado: {dir_path}")
                    return
                elif e.errno == errno.ENOTEMPTY:  # Directorio no vacío
                    self.log_message(f"Directorio no vacío, reintentando la eliminación ({i+1}/{retries}): {dir_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Error al eliminar el directorio: {dir_path} - {e}")
                    time.sleep(delay)  # Esperar incluso para otros errores
            except Exception as e:
                self.log_message(f"Error inesperado al eliminar el directorio: {dir_path} - {e}")
                return

    def setup_ffmpeg(self):
        """Verifica si FFmpeg está disponible."""
        try:
            self.run_subprocess_with_timeout(['ffmpeg', '-version'], 10, "Verificación de FFmpeg fallida")
            self.ffmpeg_available = True
            self.log_message("FFmpeg detectado.")
        except FileNotFoundError:
            self.ffmpeg_available = False
            self.log_message("FFmpeg no encontrado. Instala FFmpeg.")
            messagebox.showwarning("FFmpeg no encontrado", "Se requiere FFmpeg. Por favor, instálalo.")
        except Exception as e:
            self.ffmpeg_available = False
            self.log_message(f"Verificación de FFmpeg fallida: {e}")
            messagebox.showwarning("Error de FFmpeg", "Verificación de FFmpeg fallida. Verifica la instalación.")

    def setup_gui(self):
        """Configura la interfaz gráfica de usuario."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Encabezado ---
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        title_label = ttk.Label(header_frame, text="🎬 LinguoAI VideoVoice Pro", font=('Helvetica', 16, 'bold'))
        title_label.pack()

        # --- Selección de archivos ---
        file_frame = ttk.LabelFrame(main_frame, text="Archivos de video", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        # Usar diseño de cuadrícula para filas de entrada/salida
        source_frame = ttk.Frame(file_frame)
        source_frame.pack(fill=tk.X, pady=2)
        ttk.Label(source_frame, text="📹 Entrada:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_file, width=40)
        self.source_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(source_frame, text="Buscar", command=self.browse_source, style="Accent.TButton").grid(row=0, column=2, padx=5)

        target_frame = ttk.Frame(file_frame)
        target_frame.pack(fill=tk.X, pady=2)
        ttk.Label(target_frame, text="💾 Salida:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.target_entry = ttk.Entry(target_frame, textvariable=self.target_file, width=40)
        self.target_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(target_frame, text="Buscar", command=self.browse_target, style="Accent.TButton").grid(row=0, column=2, padx=5)

        # Hacer que las columnas de entrada sean expandibles
        source_frame.columnconfigure(1, weight=1)
        target_frame.columnconfigure(1, weight=1)


        # --- Selección de idioma ---
        lang_frame = ttk.LabelFrame(main_frame, text="Configuración de voz", padding="10")
        lang_frame.pack(fill=tk.X, pady=5)

        lang_combo_frame = ttk.Frame(lang_frame)
        lang_combo_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lang_combo_frame, text="🗣️ Idioma de destino:").pack(side=tk.LEFT, padx=5)

        # Combobox con búsqueda
        self.language_search_var = tk.StringVar()
        self.language_search_var.trace("w", self.update_language_list)
        self.language_search_entry = ttk.Entry(lang_combo_frame, textvariable=self.language_search_var)
        self.language_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.lang_combo = ttk.Combobox(
            lang_combo_frame,
            values=[f"{flag} {name}" for code, (name, flag) in self.languages.items()],
            state="readonly"
        )
        self.lang_combo.pack(side=tk.LEFT, fill=tk.X, expand=False, padx=5)
        self.lang_combo.set("🇬🇧 Inglés")  # Establecer predeterminado después de crear el combobox
        self.language_search_entry.bind("<Return>", (lambda event: self.lang_combo.focus()))
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_language_selected)

        # --- Opciones de fusión ---
        options_frame = ttk.LabelFrame(main_frame, text="Opciones de fusión", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        # Clave API de Gemini
        brain_frame = ttk.Frame(options_frame)
        brain_frame.pack(fill=tk.X, pady=2)

        # --- Descripción de la clave API de Gemini ---
        gemini_description_label = ttk.Label(
            brain_frame,
            wraplength=600,  # Ajustar el texto
            justify=tk.LEFT,  # Justificar el texto a la izquierda
            text="Este programa utiliza la API de Google Gemini Pro para el acortamiento opcional de texto.  "
                 "Esto puede ayudar a reducir la longitud general del texto traducido manteniendo la información clave.\n"
                 "Se requiere una clave API para usar esta función. Si no tienes una clave, puedes omitir este paso, "
                 "y el programa continuará sin acortar."
        )
        gemini_description_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        gemini_link_label = ttk.Label(
            brain_frame,
            text="Obtén una clave API de Gemini aquí: ",
            foreground="dodgerblue",  # Cambiado a un azul más sutil
            cursor="hand2"  # Cambiar el cursor al pasar el ratón
        )
        gemini_link_label.pack(side=tk.TOP, fill=tk.X)
        gemini_link_label.bind("<Button-1>", lambda e: self.open_webpage("https://makersuite.google.com/app/apikey"))

        self.gemini_key_label = ttk.Label(brain_frame, text="Clave API de Gemini:")
        self.gemini_key_label.pack(side=tk.LEFT, padx=5)
        self.gemini_key_entry = ttk.Entry(brain_frame, textvariable=self.gemini_key, width=30, show="*")
        self.gemini_key_entry.pack(side=tk.LEFT)

        self.start_button = ttk.Button(options_frame, text="🚀 Iniciar procesamiento", command=self.start_processing,
                                       style="Success.TButton")
        self.start_button.pack(pady=10)

        # --- Progreso ---
        progress_frame = ttk.LabelFrame(main_frame, text="Progreso", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100,
                                            style="Success.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(progress_frame, text="Listo")
        self.status_label.pack()

        # --- Registro ---
        log_frame = ttk.LabelFrame(main_frame, text="Registro de procesamiento", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, state='disabled', wrap=tk.WORD)  # Habilitar ajuste de línea
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def on_language_selected(self, event):
        """Maneja la selección de idioma desde el combobox."""
        selected_lang_str = self.lang_combo.get()
        for code, (name, flag) in self.languages.items():
            if f"{flag} {name}" == selected_lang_str:
                self.target_language = code  # Actualizar el idioma de destino
                self.log_message(f"Idioma de destino seleccionado: {name} ({code})")
                break

    def update_language_list(self, *args):
        """Filtra la lista de idiomas según la entrada de búsqueda."""
        search_term = self.language_search_var.get().lower()
        filtered_languages = [
            f"{flag} {name}"
            for code, (name, flag) in self.languages.items()
            if search_term in name.lower() or search_term in code.lower() or search_term in flag.lower()
        ]
        self.lang_combo['values'] = filtered_languages
        if filtered_languages:
            self.lang_combo.set(filtered_languages[0])  # Establecer al primer resultado
        else:
            self.lang_combo.set('')  # Limpiar si no hay resultados
    def browse_source(self):
        """Abre un diálogo de archivo para seleccionar el video fuente."""
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo de video",
            filetypes=[("Archivos de video", "*.mp4 *.avi *.mkv *.mov"), ("Todos los archivos", "*.*")]
        )
        if filename:
            self.source_file.set(filename)
            base = os.path.splitext(filename)[0]
            self.target_file.set(f"{base}_traducido.mp4")
            self.source_entry.xview_moveto(1) # Desplazar al final
            self.target_entry.xview_moveto(1)

    def browse_target(self):
        """Abre un diálogo de archivo para seleccionar la ruta del video de destino."""
        filename = filedialog.asksaveasfilename(
            title="Guardar video traducido",
            defaultextension=".mp4",
            filetypes=[("Archivos MP4", "*.mp4")]
        )
        if filename:
            self.target_file.set(filename)
            self.source_entry.xview_moveto(1)  # Desplazar al final
            self.target_entry.xview_moveto(1)
    def log_message(self, message):
        """Registra un mensaje en la GUI y en la lista de registro interna."""
        self.log_messages.append(message)
        self.log_text.config(state='normal')  # Habilitar temporalmente
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')  # Deshabilitar de nuevo
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def load_api_keys_from_environment(self):
        """Carga las claves API de las variables de entorno."""
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_key.set(self.gemini_api_key)  # Cargar en la GUI

    def start_processing(self):
        """Inicia el procesamiento de video en un hilo separado."""
        if not self.ffmpeg_available:
            messagebox.showerror("Error", "¡Se requiere FFmpeg!")
            return
        if not self.source_file.get() or not self.target_file.get():
            messagebox.showerror("Error", "Selecciona los archivos de entrada y salida.")
            return

        try:
            lang_str = self.lang_combo.get()
            self.target_language = next(
                (code for code, (name, flag) in self.languages.items() if f"{flag} {name}" == lang_str), None
            )
            if self.target_language is None:
                raise ValueError("Idioma seleccionado inválido.")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        self.gemini_api_key = self.gemini_key.get()
        if self.gemini_api_key:
            self.init_gemini()
        else:
            self.log_message("Clave API de Gemini no proporcionada. Omitiendo el acortamiento.")

        self.start_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.log_text.config(state='normal')
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state='disabled')
        self.log_messages = []
        threading.Thread(target=self.process_video, daemon=True).start()

    def load_whisper_model(self):
        """Carga el modelo Faster Whisper."""
        try:
            self.log_message(f"Cargando modelo Whisper ({self.model_size}) en {self.device}...")
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.log_message("Modelo Whisper cargado con éxito.")
        except Exception as e:
            self.log_message(f"Error al cargar el modelo Whisper: {e}")
            raise

    def extract_audio_in_chunks(self, video_path: str) -> Tuple[List[str], float]:
        """Extrae el audio del video en fragmentos."""
        self.log_message(f"Extrayendo audio de: {video_path}")
        audio_chunks = []
        try:
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Error al obtener la duración del video")
            total_duration = float(duration_output.strip())
            self.log_message(f"Duración total del video: {total_duration:.2f} segundos")

            for start_time in range(0, int(total_duration), self.audio_extraction_chunk_size):
                end_time = min(start_time + self.audio_extraction_chunk_size, total_duration)
                duration = end_time - start_time
                timestamp = int(time.time())
                chunk_dir = os.path.join(self.process_temp_dir, f"audio_chunk_{start_time}_{timestamp}")
                os.makedirs(chunk_dir, exist_ok=True)
                chunk_filename = os.path.join(chunk_dir, f"audio_chunk_{start_time}_{timestamp}.mp3")

                self.extract_audio_chunk(video_path, chunk_filename, start_time, duration)
                audio_chunks.append(chunk_filename)
            return audio_chunks, total_duration

        except Exception as e:
            self.log_message(f"Error durante la extracción de audio: {e}")
            raise  # Volver a lanzar la excepción para que se maneje en process_video

    def transcribe_audio_in_chunks(self, audio_chunks: List[str], language: str) -> Tuple[List[Dict], int]:
        """Transcribe fragmentos de audio utilizando Faster Whisper."""
        all_segments = []
        total_words = 0
        if self.whisper_model is None:
            raise ValueError("Modelo Whisper no cargado.")

        try:
            for i, chunk_path in enumerate(audio_chunks):
                self.log_message(f"Transcribiendo fragmento {i+1}/{len(audio_chunks)}: {chunk_path}")
                segments, info = self.whisper_model.transcribe(
                    chunk_path,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
                for segment in segments:
                    all_segments.append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text.strip()
                    })
                    total_words += len(segment.text.split())
                self.log_message(f"Transcripción del fragmento {i+1} completada.")
                self.root.update()  # Actualizar GUI

                self.remove_file_with_retry(chunk_path)
                self.remove_directory_with_retry(os.path.dirname(chunk_path))

            self.log_message(f"Palabras totales transcritas: {total_words}")
            return all_segments, total_words
        except Exception as e:
            self.log_message(f"Error durante la transcripción: {e}")
            raise

    def translate_and_refine_segments(self, segments: List[Dict]) -> Tuple[List[Dict], int]:
        """Traduce segmentos y maneja posibles errores de traducción."""
        translated_segments = []
        total_words_translated = 0
        translator = GoogleTranslator(source='auto', target=self.target_language)

        try:
            for segment in segments:
                self.log_message(f"Traduciendo segmento: '{segment['text'][:50]}...' a {self.target_language}")
                translated_text = translator.translate(segment['text'])
                if not isinstance(translated_text, str):
                    self.log_message(
                        f"ADVERTENCIA: translator.translate no devolvió una cadena de texto. Tipo: {type(translated_text)}, Valor: {translated_text}")
                    translated_text = ""  # Establecer a cadena vacía
                translated_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': translated_text
                })
                total_words_translated += len(translated_text.split())
                self.log_message(f"Segmento traducido: '{translated_text[:50]}...'")
            return translated_segments, total_words_translated
        except Exception as e:
            self.log_message(f"Error durante la traducción: {e}")
            raise

    def generate_tts_audio_for_segments(self, translated_segments: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
        """Genera audio TTS para un lote de segmentos traducidos."""
        tts_chunks = []
        updated_segments = []
        total_duration = 0
        try:
            for i, segment in enumerate(translated_segments):
                text = segment['text']
                self.log_message(f"Generando TTS para el segmento {i+1}/{len(translated_segments)}: '{text[:50]}...'")
                timestamp = int(time.time())
                tts_dir = os.path.join(self.process_temp_dir, f"tts_segment_dir_{i}_{timestamp}")
                os.makedirs(tts_dir, exist_ok=True)
                tts_filename = f"tts_segment_{i}_{timestamp}.mp3"
                tts_filepath = os.path.join(tts_dir, tts_filename)

                try:
                    tts = gTTS(text=text, lang=self.target_language)
                    with open(tts_filepath, 'wb') as tts_file:
                        tts.write_to_fp(tts_file)

                    tts_chunks.append(tts_filepath)
                    duration_command = [
                        "ffprobe",
                        "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        tts_filepath
                    ]
                    duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30,
                                                                        "Error al obtener la duración del segmento TTS")
                    segment_duration = float(duration_output.strip())
                    updated_segments.append({
                        'start': total_duration,
                        'end': total_duration + segment_duration,
                        'text': text,
                    })
                    total_duration += segment_duration
                except Exception as e:
                    self.log_message(f"Error al generar TTS para el segmento {i + 1}: {e}")
                    # Limpiar cualquier archivo creado en este lote hasta ahora
                    for file in tts_chunks:
                        self.remove_file_with_retry(file)
                        self.remove_directory_with_retry(os.path.dirname(file))  # Intentar eliminar el directorio
                    return None, []  # Indicar fallo

                self.log_message(f"TTS para el segmento {i+1} generado.")
                self.root.update()  # Mantener la GUI responsiva

            self.log_message(f"Fusionando {len(tts_chunks)} fragmentos TTS...")
            if tts_chunks:
                merged_audio_path = os.path.join(self.process_temp_dir, f"merged_tts_audio_{int(time.time())}.mp3")
                self.merge_audio_chunks(tts_chunks, merged_audio_path)
                self.log_message(f"Audio TTS fusionado: {merged_audio_path}")

                for tts_file in tts_chunks:
                    self.remove_file_with_retry(tts_file)
                    self.remove_directory_with_retry(os.path.dirname(tts_file))

                return merged_audio_path, updated_segments
            else:
                self.log_message("No se generaron segmentos TTS.")
                return None, []

        except Exception as e:
            self.log_message(f"Error durante la generación/fusión de TTS: {e}")
            for file in tts_chunks:  # Corregido
                self.remove_file_with_retry(file)
                self.remove_directory_with_retry(os.path.dirname(file))  # y el directorio
            return None, []

    def open_webpage(self, url):
        """Abre una página web en el navegador predeterminado."""
        import webbrowser
        webbrowser.open(url)

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Descargando tokenizador punkt de NLTK...")
        nltk.download('punkt')
    app = LinguoAIVideoVoicePro()
    app.root.mainloop()