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
        # --- Konfigurationsparameter ---
        self.ffmpeg_timeout = 30
        self.process_timeout = 7200
        self.chunk_timeout = 600
        self.validate_chunk_timeout = 20
        self.audio_extraction_chunk_size = 60
        self.transcribe_chunk_size = 60
        self.tts_batch_size = 10
        self.model_size = "medium"
        self.target_language = "en"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gemini_api_key = ""
        self.tts_engine = "gtts"
        self.hw_accel_info = self.detect_hardware_acceleration()  # Speichern der Info
        self.hw_accel = self.hw_accel_info['accel'] if self.hw_accel_info else None

        # --- Interner Status ---
        self.whisper_model: Optional[WhisperModel] = None
        self.current_process: Optional[subprocess.Popen] = None
        self.process_start_time: Optional[float] = None
        self.process_temp_dir: str = ""

        # --- GUI-Komponenten ---
        self.root = ttk.Window(themename="darkly")
        self.root.title("LinguoAI VideoVoice Pro")
        self.root.geometry("640x850")  # Set initial size

        self.source_file = tk.StringVar()
        self.target_file = tk.StringVar()
        self.selected_language = tk.StringVar(value="en")
        self.gemini_key = tk.StringVar()  # Gemini API-Schlüssel
        self.progress_var = tk.DoubleVar(value=0)
        self.log_messages: List[str] = []

        self.languages = {
            "af": ("Afrikaans", "🇿🇦"),
            "sq": ("Albanisch", "🇦🇱"),
            "am": ("Amharisch", "🇪🇹"),
            "ar": ("Arabisch", "🇸🇦"),
            "eu": ("Baskisch", "🇪🇸"),
            "bn": ("Bengalisch", "🇧🇩"),
            "bs": ("Bosnisch", "🇧🇦"),
            "bg": ("Bulgarisch", "🇧🇬"),
            "ca": ("Katalanisch", "🇦🇩"),
            "zh-CN": ("Chinesisch (Vereinfacht)", "🇨🇳"),
            "zh-TW": ("Chinesisch (Traditionell)", "🇹🇼"),
            "hr": ("Kroatisch", "🇭🇷"),
            "cs": ("Tschechisch", "🇨🇿"),
            "da": ("Dänisch", "🇩🇰"),
            "nl": ("Niederländisch", "🇳🇱"),
            "en": ("Englisch", "🇬🇧"),
            "et": ("Estnisch", "🇪🇪"),
            "tl": ("Philippinisch", "🇵🇭"),
            "fi": ("Finnisch", "🇫🇮"),
            "fr": ("Französisch", "🇫🇷"),
            "gl": ("Galizisch", "🇪🇸"),
            "de": ("Deutsch", "🇩🇪"),
            "el": ("Griechisch", "🇬🇷"),
            "gu": ("Gujarati", "🇮🇳"),
            "ha": ("Haussa", "🇳🇬"),
            "he": ("Hebräisch", "🇮🇱"),
            "hi": ("Hindi", "🇮🇳"),
            "hu": ("Ungarisch", "🇭🇺"),
            "is": ("Isländisch", "🇮🇸"),
            "id": ("Indonesisch", "🇮🇩"),
            "it": ("Italienisch", "🇮🇹"),
            "ja": ("Japanisch", "🇯🇵"),
            "jw": ("Javanisch", "🇮🇩"),
            "kn": ("Kannada", "🇮🇳"),
            "km": ("Khmer", "🇰🇭"),
            "ko": ("Koreanisch", "🇰🇷"),
            "la": ("Latein", "🇻🇦"),
            "lv": ("Lettisch", "🇱🇻"),
            "lt": ("Litauisch", "🇱🇹"),
            "ms": ("Malaiisch", "🇲🇾"),
            "mr": ("Marathi", "🇮🇳"),
            "ml": ("Malayalam", "🇮🇳"),
            "my": ("Myanmar (Burmesisch)", "🇲🇲"),
            "ne": ("Nepali", "🇳🇵"),
            "no": ("Norwegisch", "🇳🇴"),
            "pa": ("Punjabi", "🇮🇳"),
            "pl": ("Polnisch", "🇵🇱"),
            "pt": ("Portugiesisch", "🇵🇹"),
            "ro": ("Rumänisch", "🇷🇴"),
            "ru": ("Russisch", "🇷🇺"),
            "sr": ("Serbisch", "🇷🇸"),
            "si": ("Singhalesisch", "🇱🇰"),
            "sk": ("Slowakisch", "🇸🇰"),
            "sl": ("Slowenisch", "🇸🇮"),
            "es": ("Spanisch", "🇪🇸"),
            "su": ("Sunda", "🇮🇩"),
            "sw": ("Swahili", "🇰🇪"),
            "sv": ("Schwedisch", "🇸🇪"),
            "ta": ("Tamil", "🇮🇳"),
            "te": ("Telugu", "🇮🇳"),
            "th": ("Thai", "🇹🇭"),
            "tr": ("Türkisch", "🇹🇷"),
            "uk": ("Ukrainisch", "🇺🇦"),
            "ur": ("Urdu", "🇵🇰"),
            "vi": ("Vietnamesisch", "🇻🇳"),
            "cy": ("Walisisch", "🇬🇧")
        }

        # --- Initialisierung ---
        self.setup_gui()
        self.setup_ffmpeg()
        self.load_api_keys_from_environment()
        self.load_whisper_model()
        if self.gemini_api_key:
            self.init_gemini()
        self.log_hardware_acceleration()

    def seconds_to_srt_time(self, seconds: float) -> str:
        """Konvertiert Sekunden in das SRT-Zeitformat (HH:MM:SS,mmm)."""
        milliseconds = int((seconds * 1000) % 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    def create_srt_file(self, segments: List[Dict], output_path: str):
        """Erstellt eine SRT-Datei aus den transkribierten/übersetzten Segmenten."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments):
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text']

                    # Konvertieren Sie Sekunden in das SRT-Zeitformat
                    start_time_srt = self.seconds_to_srt_time(start_time)
                    end_time_srt = self.seconds_to_srt_time(end_time)

                    f.write(f"{i + 1}\n")
                    f.write(f"{start_time_srt} --> {end_time_srt}\n")
                    f.write(f"{text}\n\n")

            self.log_message(f"SRT-Datei erstellt: {output_path}")

        except Exception as e:
            self.log_message(f"Fehler beim Erstellen der SRT-Datei: {e}")

    def log_hardware_acceleration(self):
        """Protokolliert die Hardwarebeschleunigungsinformationen."""
        if self.hw_accel_info:
            self.log_message(f"Hardwarebeschleunigung erkannt: {self.hw_accel_info['accel']} ({self.hw_accel_info['info']})")
        else:
            self.log_message("Keine Hardwarebeschleunigung erkannt.")

    def detect_hardware_acceleration(self):
        """Erkennt Hardwarebeschleunigung (NVIDIA, Intel, AMD)."""
        try:
            # NVIDIA
            try:
                subprocess.run(['nvidia-smi'], check=True, capture_output=True)
                return {'accel': 'cuda', 'info': 'NVIDIA GPU detected'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # Intel (Quick Sync)
            try:
                output = subprocess.run(['vainfo'], check=True, capture_output=True, text=True).stdout
                if "VA-API" in output:  # Sehr grobe Prüfung, kann verfeinert werden
                    return {'accel': 'qsv', 'info': 'Intel Quick Sync detected'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # AMD (AMF)  - etwas komplexer, da es keine einfache "amfinfo" gibt
            #  Man könnte nach bestimmten Treibern/Geräten suchen, aber das ist OS-spezifisch.
            #  Hier eine sehr einfache, unvollständige Prüfung für Linux:
            if os.name == 'posix':  # Linux/macOS
                try:
                    output = subprocess.run(['lspci', '-v'], check=True, capture_output=True, text=True).stdout
                    if "Advanced Micro Devices, Inc. [AMD/ATI]" in output:
                        return {'accel': 'h264_vaapi', 'info': 'AMD GPU detected (VAAPI)'}  # Vermutung!
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass

            return None  # Keine Hardwarebeschleunigung gefunden

        except Exception as e:
            self.log_message(f"Fehler bei der Erkennung der Hardwarebeschleunigung: {e}")
            return None
    def init_gemini(self):
        """Initialisiert das Gemini Pro Modell."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.log_message("Gemini Pro model initialized.")
        except Exception as e:
            self.log_message(f"Error initializing Gemini Pro: {e}")
            messagebox.showerror("Gemini Error", f"Could not initialize Gemini Pro: {e}")
            self.model = None  # Set to None on failure

    def check_process_timeout(self):
        """Checks if the overall process has exceeded the maximum allowed time."""
        if self.process_start_time and time.time() - self.process_start_time > self.process_timeout:
            if self.current_process:
                self.log_message(f"Process timeout! Killing process (PID: {self.current_process.pid})...")
                self.kill_process_tree(self.current_process.pid)  # Kill process tree!
            raise TimeoutError("Process exceeded maximum allowed time")

    def kill_process_tree(self, pid):
        """Kills a process and all of its child processes."""
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):  # Get all children/grandchildren
                self.log_message(f"Killing child process: {child.pid}")
                child.kill()
            self.log_message(f"Killing parent process: {parent.pid}")
            parent.kill()
        except psutil.NoSuchProcess:
            self.log_message(f"Process with PID {pid} not found.")
        except Exception as e:
            self.log_message(f"Error killing process tree: {e}")

    def run_subprocess_with_timeout(self, command, timeout, error_message):
        """Runs a subprocess with dynamic timeout and liveness checks."""
        try:
            self.log_message(f"Running command with timeout {timeout}: {' '.join(command)}")
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.log_message(f"Process started with PID: {self.current_process.pid}")

            start_time = time.time()
            last_output_time = start_time

            stdout, stderr = self.current_process.communicate(timeout=timeout)  # Use communicate!
            retcode = self.current_process.returncode

            if retcode != 0:
                self.log_message(f"Process failed with error code {retcode}:")
                self.log_message(f"Stdout: {stdout}")
                self.log_message(f"Stderr: {stderr}")
                raise subprocess.CalledProcessError(retcode, command, stdout, stderr)

            self.log_message("Command completed successfully.")
            return stdout, stderr

        except subprocess.TimeoutExpired:
            self.log_message(f"Process timed out after {timeout} seconds")
            self.kill_process_tree(self.current_process.pid)  # Kill process tree!
            stdout, stderr = self.current_process.communicate()  # Get any remaining output
            self.log_message(f"Stdout: {stdout}")
            self.log_message(f"Stderr: {stderr}")
            raise TimeoutError(error_message)

        except Exception as e:
            self.log_message(f"An unexpected error occurred: {e}")
            if self.current_process:
                self.kill_process_tree(self.current_process.pid)  # Kill if still running
            raise
        finally:
            self.current_process = None

    def extract_audio_chunk(self, video_path, audio_path, start_time, duration):
        """Extracts a chunk of audio from the video."""
        command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-vn",  # Disable video
            "-acodec", "libmp3lame",
            "-q:a", "2",  # Good quality MP3
            "-loglevel", "error",  # Reduce verbosity
            audio_path
        ]
        if self.hw_accel:
            command.insert(1, "-hwaccel")
            command.insert(2, self.hw_accel)

        try:
            self.run_subprocess_with_timeout(command, self.chunk_timeout, "Audio extraction timed out")
            self.log_message(f"Audio chunk extracted: '{audio_path}'")
        except Exception as e:
            self.log_message(f"Error extracting audio chunk: {e}")
            raise

    def batch_segments(self, segments: List[Dict]) -> List[List[Dict]]:
        """Splits segments into smaller batches for TTS."""
        batched_segments = []
        for i in range(0, len(segments), self.tts_batch_size):
            batch = segments[i:i + self.tts_batch_size]
            batched_segments.append(batch)
        return batched_segments

    def validate_audio_chunk(self, chunk_path: str) -> bool:
        """Validates an audio chunk using ffprobe."""
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
                f"Validation of audio chunk timed out: {chunk_path}"
            )
            if stdout.strip():
                self.log_message(f"Audio chunk validated: {chunk_path}")
                return True
            else:
                self.log_message(f"Audio chunk validation failed (no duration): {chunk_path}")
                return False
        except Exception as e:
            self.log_message(f"Error validating audio chunk {chunk_path}: {e}")
            return False

    def merge_audio_chunks(self, audio_chunks, output_path):
        """Merges multiple audio chunks into a single file using FFmpeg."""
        if not audio_chunks:
            self.log_message("No audio chunks to merge.")
            return

        valid_chunks = [chunk for chunk in audio_chunks if self.validate_audio_chunk(chunk)]
        if not valid_chunks:
            self.log_message("No valid audio chunks to merge.")
            return

        timestamp = int(time.time())
        temp_dir = os.path.join(tempfile.gettempdir(), f"audio_merge_temp_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)

        list_file_path = os.path.join(temp_dir, "chunk_list.txt")
        try:
            with open(list_file_path, "w") as f:
                for chunk_path in valid_chunks:
                    abs_chunk_path = os.path.abspath(chunk_path)  # Use absolute path
                    f.write(f"file '{abs_chunk_path}'\n")

            command = [
                "ffmpeg",
                "-y",  # Overwrite
                "-f", "concat",
                "-safe", "0",  # Required for absolute paths with concat
                "-i", list_file_path,
                "-c", "copy",
                "-loglevel", "error",
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            # Calculate a dynamic timeout based on the number of chunks.
            merge_timeout = len(valid_chunks) * 10 + 30  # 10 seconds per chunk + 30 base
            self.run_subprocess_with_timeout(command, merge_timeout, "Audio merging timed out")
            self.log_message(f"Audio chunks merged: '{output_path}'")

        except Exception as e:
            self.log_message(f"Error merging audio chunks: {e}")
            raise
        finally:
            self.remove_file_with_retry(list_file_path)  # Use the retry function
            self.remove_directory_with_retry(temp_dir)  # and for the directory

    def merge_video_audio(self, audio_file):
        """Merges the final audio with the original video."""
        try:
            output_path = self.target_file.get()
            video_path = self.source_file.get()

            # Get video duration using ffprobe
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Failed to get video duration")
            total_duration = float(duration_output.strip())
            self.log_message(f"Video duration for merge: {total_duration:.2f} seconds")

            # Calculate dynamic timeout (e.g., 3x duration + 120 seconds)
            dynamic_timeout = int(3 * total_duration + 120)
            self.log_message(f"Dynamic timeout for merge: {dynamic_timeout} seconds")

            command = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-i', video_path,
                '-i', audio_file,
                '-c:v', 'h264_nvenc' if self.hw_accel == 'cuda' else 'libx264',  # H.265
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',  # Finish encoding when the shortest stream ends
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            self.run_subprocess_with_timeout(command, dynamic_timeout, "Video and audio merging timed out")
            self.log_message(f"Video and audio merged: {output_path}")
        except Exception as e:
            self.log_message(f"Error merging video and audio: {e}")
            raise

    def shorten_text_with_gemini(self, text: str) -> str:
        """Shortens text using Gemini Pro, handling errors and rate limits."""
        if self.model is None:
            self.log_message("Gemini Pro model not initialized. Skipping shortening.")
            return text

        try:
            prompt = f"Please shorten the following text while preserving the key information:\n\n{text}"
            time.sleep(1.5)  # Rate limiting: Pause for 1.5 seconds
            response = self.model.generate_content(prompt)
            if response and response.text:
                shortened_text = response.text
                self.log_message("Text shortened with Gemini Pro.")
                return shortened_text
            else:
                self.log_message("Gemini Pro returned empty response. Using original text.")
                return text
        except Exception as e:
            self.log_message(f"Error shortening text with Gemini: {e}")
            return text

    def process_video(self):
        """Main video processing workflow."""
        self.process_start_time = time.time()
        timestamp = int(time.time())
        self.process_temp_dir = os.path.join(tempfile.gettempdir(), f"process_temp_{timestamp}")
        os.makedirs(self.process_temp_dir, exist_ok=True)

        try:
            self.log_message("Starting video processing...")
            self.progress_var.set(5)

            # Set up the timeout check *before* starting any tasks.
            def check_timeout():
                self.check_process_timeout()
                self.root.after(1000, check_timeout)  # Check every second

            check_timeout()  # Start the timeout checker

            self.progress_var.set(10)
            audio_chunks, total_duration = self.extract_audio_in_chunks(self.source_file.get())
            self.progress_var.set(25)

            segments, total_words_original = self.transcribe_audio_in_chunks(audio_chunks, self.target_language)
            if not segments:
                raise Exception("Transcription failed.")
            self.progress_var.set(45)

            translated_segments, total_words_translated = self.translate_and_refine_segments(segments)

            # --- Gemini Text Shortening (Optional) ---
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
                    self.root.update()  # Update GUI
                translated_segments = shortened_segments

                shortening_change = (
                    ((total_words_translated - total_words_shortened) / total_words_translated) * 100
                    if total_words_translated else 0
                )
                self.log_message(f"Shortening reduced word count by: {shortening_change:.2f}%")
            else:
                total_words_shortened = total_words_translated
                self.log_message("Gemini API key not provided. Skipping shortening.")

            self.progress_var.set(60)

            # Erstellen der SRT-Datei (Beispiel)
            srt_output_path = os.path.splitext(self.target_file.get())[0] + ".srt"  # Passend zum Ausgabevideonamen
            self.create_srt_file(translated_segments, srt_output_path)

            # --- Word Count Reporting ---
            if total_words_original > 0:
                translation_change = (total_words_translated / total_words_original) * 100
                self.log_message(f"Original word count: {total_words_original}")
                self.log_message(f"Translated word count: {total_words_translated}")
                self.log_message(f"Translation word count change: {translation_change:.2f}%")
            else:
                self.log_message("Original word count is zero. Skipping percentage.")

            # --- TTS and Audio Merging ---
            batched_translated_segments = self.batch_segments(translated_segments)
            all_updated_segments = []
            merged_audio_files = []

            for i, batch in enumerate(batched_translated_segments):
                self.log_message(f"Processing TTS batch {i+1}/{len(batched_translated_segments)}")
                merged_audio_path, updated_segments = self.generate_tts_audio_for_segments(batch)
                if merged_audio_path:
                    all_updated_segments.extend(updated_segments)
                    merged_audio_files.append(merged_audio_path)
                else:
                    self.log_message(f"TTS batch {i+1} failed.")
                    # Clean up *all* previously created TTS files on failure
                    for file_path in merged_audio_files:
                        self.remove_file_with_retry(file_path)
                    raise Exception(f"TTS audio generation failed for batch {i + 1}.")

            # --- Final Audio Merge (if multiple batches) ---
            if len(merged_audio_files) > 1:
                final_merged_audio_path = os.path.join(self.process_temp_dir,
                                                        f"final_merged_tts_audio_{timestamp}.mp3")
                self.merge_audio_chunks(merged_audio_files, final_merged_audio_path)
                self.log_message(f"Final merged TTS audio: {final_merged_audio_path}")
            elif merged_audio_files:
                final_merged_audio_path = merged_audio_files[0]
                self.log_message(f"Final merged TTS audio (single batch): {final_merged_audio_path}")
            else:
                raise Exception("No TTS audio generated.")

            # --- Clean up intermediate TTS files ---
            for file_path in merged_audio_files:
                if file_path != final_merged_audio_path:  # Don't delete the final file!
                    self.remove_file_with_retry(file_path)

            self.progress_var.set(80)

            # --- Final Video/Audio Merge ---
            self.merge_video_audio(final_merged_audio_path)
            self.progress_var.set(95)

            self.log_message("Processing completed successfully! 🎉")
            messagebox.showinfo("Success", "Video processing completed!")


        except TimeoutError as e:
            self.log_message(f"Processing timed out: {str(e)}")
            messagebox.showerror("Error", f"Processing timed out: {str(e)}")
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            # --- Cleanup ---
            self.progress_var.set(0)
            self.process_start_time = None
            self.current_process = None
            self.start_button.config(state=tk.NORMAL)  # Re-enable button
            self.remove_directory_with_retry(self.process_temp_dir)

    def remove_file_with_retry(self, file_path, retries=3, delay=0.5):
        """Removes a file, retrying if necessary."""
        file_path = os.path.abspath(file_path)  # Use absolute path
        for i in range(retries):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                self.log_message(f"File removed: {file_path}")
                return  # Success
            except OSError as e:
                if e.errno == errno.ENOENT:  # File not found - already gone
                    self.log_message(f"File not found (already removed): {file_path}")
                    return
                if i < retries - 1:  # Don't wait on the last attempt
                    self.log_message(f"Retrying file removal ({i+1}/{retries}): {file_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Error removing file after multiple retries: {file_path} - {e}")
            except Exception as e:
                self.log_message(f"Unexpected error removing file: {file_path} - {e}")
                return  # Don't retry for unexpected errors

    def remove_directory_with_retry(self, dir_path, retries=5, delay=1):
        """Removes a directory, retrying if necessary (especially for non-empty)."""
        dir_path = os.path.abspath(dir_path)
        for i in range(retries):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                self.log_message(f"Directory removed: {dir_path}")
                return
            except OSError as e:
                if e.errno == errno.ENOENT:  # Dir already removed
                    self.log_message(f"Directory already removed: {dir_path}")
                    return
                elif e.errno == errno.ENOTEMPTY:  # Directory not empty
                    self.log_message(f"Directory not empty, retrying removal ({i+1}/{retries}): {dir_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Error removing directory: {dir_path} - {e}")
                    time.sleep(delay)  # Wait even for other errors
            except Exception as e:
                self.log_message(f"Unexpected error removing directory: {dir_path} - {e}")
                return

    def setup_ffmpeg(self):
        """Checks if FFmpeg is available."""
        try:
            self.run_subprocess_with_timeout(['ffmpeg', '-version'], 10, "FFmpeg check failed")
            self.ffmpeg_available = True
            self.log_message("FFmpeg detected.")
        except FileNotFoundError:
            self.ffmpeg_available = False
            self.log_message("FFmpeg not found. Install FFmpeg.")
            messagebox.showwarning("FFmpeg Missing", "FFmpeg is required.  Please install it.")
        except Exception as e:
            self.ffmpeg_available = False
            self.log_message(f"FFmpeg check failed: {e}")
            messagebox.showwarning("FFmpeg Error", "FFmpeg check failed. Check installation.")

    def setup_gui(self):
        """Sets up the graphical user interface."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Header ---
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        title_label = ttk.Label(header_frame, text="🎬 LinguoAI VideoVoice Pro", font=('Helvetica', 16, 'bold'))
        title_label.pack()

        # --- File Selection ---
        file_frame = ttk.LabelFrame(main_frame, text="Video Files", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        # Use grid layout for input/output rows
        source_frame = ttk.Frame(file_frame)
        source_frame.pack(fill=tk.X, pady=2)
        ttk.Label(source_frame, text="📹 Input:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_file, width=40)
        self.source_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(source_frame, text="Browse", command=self.browse_source, style="Accent.TButton").grid(row=0, column=2, padx=5)

        target_frame = ttk.Frame(file_frame)
        target_frame.pack(fill=tk.X, pady=2)
        ttk.Label(target_frame, text="💾 Output:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.target_entry = ttk.Entry(target_frame, textvariable=self.target_file, width=40)
        self.target_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(target_frame, text="Browse", command=self.browse_target, style="Accent.TButton").grid(row=0, column=2, padx=5)

        # Make the entry columns expandable
        source_frame.columnconfigure(1, weight=1)
        target_frame.columnconfigure(1, weight=1)


        # --- Language Selection ---
        lang_frame = ttk.LabelFrame(main_frame, text="Voice Settings", padding="10")
        lang_frame.pack(fill=tk.X, pady=5)

        lang_combo_frame = ttk.Frame(lang_frame)
        lang_combo_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lang_combo_frame, text="🗣️ Target Language:").pack(side=tk.LEFT, padx=5)

        # Searchable Combobox
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
        self.lang_combo.set("🇬🇧 English")  # Set default after combobox is created
        self.language_search_entry.bind("<Return>", (lambda event: self.lang_combo.focus()))
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_language_selected)

        # --- Merge Options ---
        options_frame = ttk.LabelFrame(main_frame, text="Merge Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        # Gemini API Key
        brain_frame = ttk.Frame(options_frame)
        brain_frame.pack(fill=tk.X, pady=2)

        # --- Gemini API Key Description ---
        gemini_description_label = ttk.Label(
            brain_frame,
            wraplength=600,  # Wrap the text
            justify=tk.LEFT,  # Left-justify the text
            text="This program uses the Google Gemini Pro API for optional text shortening.  "
                 "This can help reduce the overall length of the translated text while preserving key information.\n"
                 "An API key is required to use this feature.  If you don't have a key, you can skip this step, "
                 "and the program will proceed without shortening."
        )
        gemini_description_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        gemini_link_label = ttk.Label(
            brain_frame,
            text="Get a Gemini API key here: ",
            foreground="dodgerblue",  # Changed to a more subtle blue
            cursor="hand2"  # Change cursor on hover
        )
        gemini_link_label.pack(side=tk.TOP, fill=tk.X)
        gemini_link_label.bind("<Button-1>", lambda e: self.open_webpage("https://makersuite.google.com/app/apikey"))

        self.gemini_key_label = ttk.Label(brain_frame, text="Gemini API Key:")
        self.gemini_key_label.pack(side=tk.LEFT, padx=5)
        self.gemini_key_entry = ttk.Entry(brain_frame, textvariable=self.gemini_key, width=30, show="*")
        self.gemini_key_entry.pack(side=tk.LEFT)

        self.start_button = ttk.Button(options_frame, text="🚀 Start Processing", command=self.start_processing,
                                       style="Success.TButton")
        self.start_button.pack(pady=10)

        # --- Progress ---
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100,
                                            style="Success.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack()

        # --- Log ---
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, state='disabled', wrap=tk.WORD)  # Enable word wrapping
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def on_language_selected(self, event):
        """Handles language selection from combobox."""
        selected_lang_str = self.lang_combo.get()
        for code, (name, flag) in self.languages.items():
            if f"{flag} {name}" == selected_lang_str:
                self.target_language = code  # Update the target language
                self.log_message(f"Selected target language: {name} ({code})")
                break

    def update_language_list(self, *args):
        """Filters the language list based on the search input."""
        search_term = self.language_search_var.get().lower()
        filtered_languages = [
            f"{flag} {name}"
            for code, (name, flag) in self.languages.items()
            if search_term in name.lower() or search_term in code.lower() or search_term in flag.lower()
        ]
        self.lang_combo['values'] = filtered_languages
        if filtered_languages:
            self.lang_combo.set(filtered_languages[0])  # Set to the first match
        else:
            self.lang_combo.set('')  # Clear if no matches
    def browse_source(self):
        """Opens a file dialog to select the source video."""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
        )
        if filename:
            self.source_file.set(filename)
            base = os.path.splitext(filename)[0]
            self.target_file.set(f"{base}_translated.mp4")
            self.source_entry.xview_moveto(1) #Scroll to the end
            self.target_entry.xview_moveto(1)

    def browse_target(self):
        """Opens a file dialog to select the target video path."""
        filename = filedialog.asksaveasfilename(
            title="Save Translated Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")]
        )
        if filename:
            self.target_file.set(filename)
            self.source_entry.xview_moveto(1)  # Scroll to the end
            self.target_entry.xview_moveto(1)
    def log_message(self, message):
        """Logs a message to the GUI and the internal log list."""
        self.log_messages.append(message)
        self.log_text.config(state='normal')  # Temporarily enable
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')  # Disable again
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def load_api_keys_from_environment(self):
        """Loads API keys from environment variables."""
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_key.set(self.gemini_api_key)  # Load into GUI

    def start_processing(self):
        """Starts the video processing in a separate thread."""
        if not self.ffmpeg_available:
            messagebox.showerror("Error", "FFmpeg is required!")
            return
        if not self.source_file.get() or not self.target_file.get():
            messagebox.showerror("Error", "Select input and output files.")
            return

        try:
            lang_str = self.lang_combo.get()
            self.target_language = next(
                (code for code, (name, flag) in self.languages.items() if f"{flag} {name}" == lang_str), None
            )
            if self.target_language is None:
                raise ValueError("Invalid language selected.")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        self.gemini_api_key = self.gemini_key.get()
        if self.gemini_api_key:
            self.init_gemini()
        else:
            self.log_message("Gemini API key not provided. Skipping shortening.")

        self.start_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.log_text.config(state='normal')
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state='disabled')
        self.log_messages = []
        threading.Thread(target=self.process_video, daemon=True).start()

    def load_whisper_model(self):
        """Loads the Faster Whisper model."""
        try:
            self.log_message(f"Loading Whisper model ({self.model_size}) on {self.device}...")
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.log_message("Whisper model loaded successfully.")
        except Exception as e:
            self.log_message(f"Error loading Whisper model: {e}")
            raise

    def extract_audio_in_chunks(self, video_path: str) -> Tuple[List[str], float]:
        """Extracts audio from the video in chunks."""
        self.log_message(f"Extracting audio from: {video_path}")
        audio_chunks = []
        try:
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Failed to get video duration")
            total_duration = float(duration_output.strip())
            self.log_message(f"Total video duration: {total_duration:.2f} seconds")

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
            self.log_message(f"Error during audio extraction: {e}")
            raise  # Re-raise the exception to be handled in process_video

    def transcribe_audio_in_chunks(self, audio_chunks: List[str], language: str) -> Tuple[List[Dict], int]:
        """Transcribes audio chunks using Faster Whisper."""
        all_segments = []
        total_words = 0
        if self.whisper_model is None:
            raise ValueError("Whisper model not loaded.")

        try:
            for i, chunk_path in enumerate(audio_chunks):
                self.log_message(f"Transcribing chunk {i+1}/{len(audio_chunks)}: {chunk_path}")
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
                self.log_message(f"Transcription of chunk {i+1} complete.")
                self.root.update()  # Update GUI

                self.remove_file_with_retry(chunk_path)
                self.remove_directory_with_retry(os.path.dirname(chunk_path))

            self.log_message(f"Total words transcribed: {total_words}")
            return all_segments, total_words
        except Exception as e:
            self.log_message(f"Error during transcription: {e}")
            raise

    def translate_and_refine_segments(self, segments: List[Dict]) -> Tuple[List[Dict], int]:
        """Translates segments and handles potential translation errors."""
        translated_segments = []
        total_words_translated = 0
        translator = GoogleTranslator(source='auto', target=self.target_language)

        try:
            for segment in segments:
                self.log_message(f"Translating segment: '{segment['text'][:50]}...' to {self.target_language}")
                translated_text = translator.translate(segment['text'])
                if not isinstance(translated_text, str):
                    self.log_message(
                        f"WARNING: translator.translate did not return a string. Type: {type(translated_text)}, Value: {translated_text}")
                    translated_text = ""  # Set to empty string
                translated_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': translated_text
                })
                total_words_translated += len(translated_text.split())
                self.log_message(f"Translated segment: '{translated_text[:50]}...'")
            return translated_segments, total_words_translated
        except Exception as e:
            self.log_message(f"Error during translation: {e}")
            raise

    def generate_tts_audio_for_segments(self, translated_segments: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
        """Generates TTS audio for a batch of translated segments."""
        tts_chunks = []
        updated_segments = []
        total_duration = 0
        try:
            for i, segment in enumerate(translated_segments):
                text = segment['text']
                self.log_message(f"Generating TTS for segment {i+1}/{len(translated_segments)}: '{text[:50]}...'")
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
                                                                        "Failed to get TTS segment duration")
                    segment_duration = float(duration_output.strip())
                    updated_segments.append({
                        'start': total_duration,
                        'end': total_duration + segment_duration,
                        'text': text,
                    })
                    total_duration += segment_duration
                except Exception as e:
                    self.log_message(f"Error generating TTS for segment {i + 1}: {e}")
                    # Clean up any created files in this batch so far
                    for file in tts_chunks:
                        self.remove_file_with_retry(file)
                        self.remove_directory_with_retry(os.path.dirname(file))  # Try to remove dir
                    return None, []  # Indicate failure

                self.log_message(f"TTS for segment {i+1} generated.")
                self.root.update()  # Keep GUI responsive

            self.log_message(f"Merging {len(tts_chunks)} TTS chunks...")
            if tts_chunks:
                merged_audio_path = os.path.join(self.process_temp_dir, f"merged_tts_audio_{int(time.time())}.mp3")
                self.merge_audio_chunks(tts_chunks, merged_audio_path)
                self.log_message(f"Merged TTS audio: {merged_audio_path}")

                for tts_file in tts_chunks:
                    self.remove_file_with_retry(tts_file)
                    self.remove_directory_with_retry(os.path.dirname(tts_file))

                return merged_audio_path, updated_segments
            else:
                self.log_message("No TTS segments generated.")
                return None, []

        except Exception as e:
            self.log_message(f"Error during TTS generation/merging: {e}")
            for file in tts_chunks:  # Corrected
                self.remove_file_with_retry(file)
                self.remove_directory_with_retry(os.path.dirname(file))  # and dir
            return None, []

    def open_webpage(self, url):
        """Opens a webpage in the default browser."""
        import webbrowser
        webbrowser.open(url)

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    app = LinguoAIVideoVoicePro()
    app.root.mainloop()