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
        # Initialgröße festlegen

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
            self.log_message("Gemini Pro Modell initialisiert.")
        except Exception as e:
            self.log_message(f"Fehler beim Initialisieren von Gemini Pro: {e}")
            messagebox.showerror("Gemini Fehler", f"Konnte Gemini Pro nicht initialisieren: {e}")
            self.model = None  # Set to None on failure
            # Bei Fehler auf None setzen

    def check_process_timeout(self):
        """Überprüft, ob die Gesamtprozesszeit das maximal zulässige Zeitlimit überschritten hat."""
        if self.process_start_time and time.time() - self.process_start_time > self.process_timeout:
            if self.current_process:
                self.log_message(f"Prozess-Timeout! Prozess wird beendet (PID: {self.current_process.pid})...")
                self.kill_process_tree(self.current_process.pid)  # Kill process tree!
                # Prozessbaum beenden!
            raise TimeoutError("Prozess hat das maximal zulässige Zeitlimit überschritten")

    def kill_process_tree(self, pid):
        """Beendet einen Prozess und alle seine Kindprozesse."""
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):  # Get all children/grandchildren
                # Alle Kinder/Enkelkinder holen
                self.log_message(f"Beende Kindprozess: {child.pid}")
                child.kill()
            self.log_message(f"Beende Elternprozess: {parent.pid}")
            parent.kill()
        except psutil.NoSuchProcess:
            self.log_message(f"Prozess mit PID {pid} nicht gefunden.")
        except Exception as e:
            self.log_message(f"Fehler beim Beenden des Prozessbaums: {e}")

    def run_subprocess_with_timeout(self, command, timeout, error_message):
        """Führt einen Subprozess mit dynamischem Timeout und Liveness-Checks aus."""
        try:
            self.log_message(f"Führe Befehl mit Timeout {timeout} aus: {' '.join(command)}")
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.log_message(f"Prozess mit PID gestartet: {self.current_process.pid}")

            start_time = time.time()
            last_output_time = start_time

            stdout, stderr = self.current_process.communicate(timeout=timeout)  # Use communicate!
            # Communicate verwenden!
            retcode = self.current_process.returncode

            if retcode != 0:
                self.log_message(f"Prozess fehlgeschlagen mit Fehlercode {retcode}:")
                self.log_message(f"Stdout: {stdout}")
                self.log_message(f"Stderr: {stderr}")
                raise subprocess.CalledProcessError(retcode, command, stdout, stderr)

            self.log_message("Befehl erfolgreich abgeschlossen.")
            return stdout, stderr

        except subprocess.TimeoutExpired:
            self.log_message(f"Prozess-Timeout nach {timeout} Sekunden")
            self.kill_process_tree(self.current_process.pid)  # Kill process tree!
            # Prozessbaum beenden!
            stdout, stderr = self.current_process.communicate()  # Get any remaining output
            # Jegliche verbleibende Ausgabe holen
            self.log_message(f"Stdout: {stdout}")
            self.log_message(f"Stderr: {stderr}")
            raise TimeoutError(error_message)

        except Exception as e:
            self.log_message(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
            if self.current_process:
                self.kill_process_tree(self.current_process.pid)  # Kill if still running
                # Beenden, falls noch läuft
            raise
        finally:
            self.current_process = None

    def extract_audio_chunk(self, video_path, audio_path, start_time, duration):
        """Extrahiert einen Audio-Chunk aus dem Video."""
        command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            # Ausgabedateien ohne Nachfrage überschreiben
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-vn",  # Disable video
            # Video deaktivieren
            "-acodec", "libmp3lame",
            "-q:a", "2",  # Good quality MP3
            # Gute MP3-Qualität
            "-loglevel", "error",  # Reduce verbosity
            # Ausführlichkeit reduzieren
            audio_path
        ]
        if self.hw_accel:
            command.insert(1, "-hwaccel")
            command.insert(2, self.hw_accel)

        try:
            self.run_subprocess_with_timeout(command, self.chunk_timeout, "Audio-Extraktion Timeout")
            self.log_message(f"Audio-Chunk extrahiert: '{audio_path}'")
        except Exception as e:
            self.log_message(f"Fehler beim Extrahieren des Audio-Chunks: {e}")
            raise

    def batch_segments(self, segments: List[Dict]) -> List[List[Dict]]:
        """Teilt Segmente in kleinere Batches für TTS auf."""
        batched_segments = []
        for i in range(0, len(segments), self.tts_batch_size):
            batch = segments[i:i + self.tts_batch_size]
            batched_segments.append(batch)
        return batched_segments

    def validate_audio_chunk(self, chunk_path: str) -> bool:
        """Validiert einen Audio-Chunk mit ffprobe."""
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
                f"Validierung des Audio-Chunks Timeout: {chunk_path}"
            )
            if stdout.strip():
                self.log_message(f"Audio-Chunk validiert: {chunk_path}")
                return True
            else:
                self.log_message(f"Audio-Chunk Validierung fehlgeschlagen (keine Dauer): {chunk_path}")
                return False
        except Exception as e:
            self.log_message(f"Fehler beim Validieren des Audio-Chunks {chunk_path}: {e}")
            return False

    def merge_audio_chunks(self, audio_chunks, output_path):
        """Führt mehrere Audio-Chunks zu einer einzigen Datei mit FFmpeg zusammen."""
        if not audio_chunks:
            self.log_message("Keine Audio-Chunks zum Zusammenführen.")
            return

        valid_chunks = [chunk for chunk in audio_chunks if self.validate_audio_chunk(chunk)]
        if not valid_chunks:
            self.log_message("Keine gültigen Audio-Chunks zum Zusammenführen.")
            return

        timestamp = int(time.time())
        temp_dir = os.path.join(tempfile.gettempdir(), f"audio_merge_temp_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)

        list_file_path = os.path.join(temp_dir, "chunk_list.txt")
        try:
            with open(list_file_path, "w") as f:
                for chunk_path in valid_chunks:
                    abs_chunk_path = os.path.abspath(chunk_path)  # Use absolute path
                    # Absoluten Pfad verwenden
                    f.write(f"file '{abs_chunk_path}'\n")

            command = [
                "ffmpeg",
                "-y",  # Overwrite
                # Überschreiben
                "-f", "concat",
                "-safe", "0",  # Required for absolute paths with concat
                # Für absolute Pfade mit concat erforderlich
                "-i", list_file_path,
                "-c", "copy",
                "-loglevel", "error",
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            # Calculate a dynamic timeout based on the number of chunks.
            # Dynamisches Timeout basierend auf der Anzahl der Chunks berechnen.
            merge_timeout = len(valid_chunks) * 10 + 30  # 10 seconds per chunk + 30 base
            # 10 Sekunden pro Chunk + 30 Basis
            self.run_subprocess_with_timeout(command, merge_timeout, "Audio-Zusammenführung Timeout")
            self.log_message(f"Audio-Chunks zusammengeführt: '{output_path}'")

        except Exception as e:
            self.log_message(f"Fehler beim Zusammenführen der Audio-Chunks: {e}")
            raise
        finally:
            self.remove_file_with_retry(list_file_path)  # Use the retry function
            # Retry-Funktion verwenden
            self.remove_directory_with_retry(temp_dir)  # and for the directory
            # und für das Verzeichnis

    def merge_video_audio(self, audio_file):
        """Führt das finale Audio mit dem Originalvideo zusammen."""
        try:
            output_path = self.target_file.get()
            video_path = self.source_file.get()

            # Get video duration using ffprobe
            # Videodauer mit ffprobe ermitteln
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Fehler beim Abrufen der Videodauer")
            total_duration = float(duration_output.strip())
            self.log_message(f"Videodauer für Zusammenführung: {total_duration:.2f} Sekunden")

            # Calculate dynamic timeout (e.g., 3x duration + 120 seconds)
            # Dynamisches Timeout berechnen (z.B. 3x Dauer + 120 Sekunden)
            dynamic_timeout = int(3 * total_duration + 120)
            self.log_message(f"Dynamisches Timeout für Zusammenführung: {dynamic_timeout} Sekunden")

            command = [
                'ffmpeg',
                '-y',  # Overwrite output file
                # Ausgabedatei überschreiben
                '-i', video_path,
                '-i', audio_file,
                '-c:v', 'h264_nvenc' if self.hw_accel == 'cuda' else 'libx264',  # H.265
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',  # Finish encoding when the shortest stream ends
                # Kodierung beenden, wenn der kürzeste Stream endet
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            self.run_subprocess_with_timeout(command, dynamic_timeout, "Video- und Audio-Zusammenführung Timeout")
            self.log_message(f"Video und Audio zusammengeführt: {output_path}")
        except Exception as e:
            self.log_message(f"Fehler beim Zusammenführen von Video und Audio: {e}")
            raise

    def shorten_text_with_gemini(self, text: str) -> str:
        """Kürzt Text mit Gemini Pro, behandelt Fehler und Ratenbegrenzungen."""
        if self.model is None:
            self.log_message("Gemini Pro Modell nicht initialisiert. Überspringe Kürzung.")
            return text

        try:
            prompt = f"Bitte kürze den folgenden Text und erhalte dabei die wichtigsten Informationen:\n\n{text}"
            time.sleep(1.5)  # Rate limiting: Pause for 1.5 seconds
            # Ratenbegrenzung: 1,5 Sekunden Pause
            response = self.model.generate_content(prompt)
            if response and response.text:
                shortened_text = response.text
                self.log_message("Text mit Gemini Pro gekürzt.")
                return shortened_text
            else:
                self.log_message("Gemini Pro hat eine leere Antwort zurückgegeben. Verwende Originaltext.")
                return text
        except Exception as e:
            self.log_message(f"Fehler beim Kürzen des Texts mit Gemini: {e}")
            return text

    def process_video(self):
        """Haupt-Workflow für die Videoverarbeitung."""
        self.process_start_time = time.time()
        timestamp = int(time.time())
        self.process_temp_dir = os.path.join(tempfile.gettempdir(), f"process_temp_{timestamp}")
        os.makedirs(self.process_temp_dir, exist_ok=True)

        try:
            self.log_message("Starte Videoverarbeitung...")
            self.progress_var.set(5)

            # Set up the timeout check *before* starting any tasks.
            # Timeout-Prüfung *vor* dem Start jeglicher Aufgaben einrichten.
            def check_timeout():
                self.check_process_timeout()
                self.root.after(1000, check_timeout)  # Check every second
                # Jede Sekunde prüfen

            check_timeout()  # Start the timeout checker
            # Timeout-Prüfung starten

            self.progress_var.set(10)
            audio_chunks, total_duration = self.extract_audio_in_chunks(self.source_file.get())
            self.progress_var.set(25)

            segments, total_words_original = self.transcribe_audio_in_chunks(audio_chunks, self.target_language)
            if not segments:
                raise Exception("Transkription fehlgeschlagen.")
            self.progress_var.set(45)

            translated_segments, total_words_translated = self.translate_and_refine_segments(segments)

            # --- Gemini Text Shortening (Optional) ---
            # --- Gemini Textkürzung (Optional) ---
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
                    # GUI aktualisieren
                translated_segments = shortened_segments

                shortening_change = (
                    ((total_words_translated - total_words_shortened) / total_words_translated) * 100
                    if total_words_translated else 0
                )
                self.log_message(f"Kürzung hat die Wortanzahl um {shortening_change:.2f}% reduziert")
            else:
                total_words_shortened = total_words_translated
                self.log_message("Gemini API-Schlüssel nicht angegeben. Überspringe Kürzung.")

            self.progress_var.set(60)

            # Erstellen der SRT-Datei (Beispiel)
            srt_output_path = os.path.splitext(self.target_file.get())[0] + ".srt"  # Passend zum Ausgabevideonamen
            # Passend zum Ausgabevideonamen
            self.create_srt_file(translated_segments, srt_output_path)

            # --- Word Count Reporting ---
            # --- Wortzählung Bericht ---
            if total_words_original > 0:
                translation_change = (total_words_translated / total_words_original) * 100
                self.log_message(f"Originale Wortanzahl: {total_words_original}")
                self.log_message(f"Übersetzte Wortanzahl: {total_words_translated}")
                self.log_message(f"Änderung der Wortanzahl durch Übersetzung: {translation_change:.2f}%")
            else:
                self.log_message("Originale Wortanzahl ist Null. Überspringe Prozentangabe.")

            # --- TTS and Audio Merging ---
            # --- TTS und Audio-Zusammenführung ---
            batched_translated_segments = self.batch_segments(translated_segments)
            all_updated_segments = []
            merged_audio_files = []

            for i, batch in enumerate(batched_translated_segments):
                self.log_message(f"Verarbeite TTS Batch {i+1}/{len(batched_translated_segments)}")
                merged_audio_path, updated_segments = self.generate_tts_audio_for_segments(batch)
                if merged_audio_path:
                    all_updated_segments.extend(updated_segments)
                    merged_audio_files.append(merged_audio_path)
                else:
                    self.log_message(f"TTS Batch {i+1} fehlgeschlagen.")
                    # Clean up *all* previously created TTS files on failure
                    # *Alle* zuvor erstellten TTS-Dateien bei Fehler löschen
                    for file_path in merged_audio_files:
                        self.remove_file_with_retry(file_path)
                    raise Exception(f"TTS-Audio-Generierung für Batch {i + 1} fehlgeschlagen.")

            # --- Final Audio Merge (if multiple batches) ---
            # --- Finale Audio-Zusammenführung (wenn mehrere Batches) ---
            if len(merged_audio_files) > 1:
                final_merged_audio_path = os.path.join(self.process_temp_dir,
                                                        f"final_merged_tts_audio_{timestamp}.mp3")
                self.merge_audio_chunks(merged_audio_files, final_merged_audio_path)
                self.log_message(f"Finales zusammengeführtes TTS-Audio: {final_merged_audio_path}")
            elif merged_audio_files:
                final_merged_audio_path = merged_audio_files[0]
                self.log_message(f"Finales zusammengeführtes TTS-Audio (einzelner Batch): {final_merged_audio_path}")
            else:
                raise Exception("Kein TTS-Audio generiert.")

            # --- Clean up intermediate TTS files ---
            # --- Zwischen-TTS-Dateien aufräumen ---
            for file_path in merged_audio_files:
                if file_path != final_merged_audio_path:  # Don't delete the final file!
                    # Finale Datei nicht löschen!
                    self.remove_file_with_retry(file_path)

            self.progress_var.set(80)

            # --- Final Video/Audio Merge ---
            # --- Finale Video/Audio-Zusammenführung ---
            self.merge_video_audio(final_merged_audio_path)
            self.progress_var.set(95)

            self.log_message("Verarbeitung erfolgreich abgeschlossen! 🎉")
            messagebox.showinfo("Erfolg", "Videoverarbeitung abgeschlossen!")


        except TimeoutError as e:
            self.log_message(f"Verarbeitung Timeout: {str(e)}")
            messagebox.showerror("Fehler", f"Verarbeitung Timeout: {str(e)}")
        except Exception as e:
            self.log_message(f"Fehler: {str(e)}")
            messagebox.showerror("Fehler", f"Verarbeitung fehlgeschlagen: {str(e)}")
        finally:
            # --- Cleanup ---
            # --- Aufräumen ---
            self.progress_var.set(0)
            self.process_start_time = None
            self.current_process = None
            self.start_button.config(state=tk.NORMAL)  # Re-enable button
            # Button wieder aktivieren
            self.remove_directory_with_retry(self.process_temp_dir)

    def remove_file_with_retry(self, file_path, retries=3, delay=0.5):
        """Entfernt eine Datei und versucht es bei Bedarf erneut."""
        file_path = os.path.abspath(file_path)  # Use absolute path
        # Absoluten Pfad verwenden
        for i in range(retries):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                self.log_message(f"Datei entfernt: {file_path}")
                return  # Success
                # Erfolg
            except OSError as e:
                if e.errno == errno.ENOENT:  # File not found - already gone
                    # Datei nicht gefunden - bereits weg
                    self.log_message(f"Datei nicht gefunden (bereits entfernt): {file_path}")
                    return
                if i < retries - 1:  # Don't wait on the last attempt
                    # Bei letztem Versuch nicht warten
                    self.log_message(f"Datei-Entfernung erneut versuchen ({i+1}/{retries}): {file_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Fehler beim Entfernen der Datei nach mehreren Versuchen: {file_path} - {e}")
            except Exception as e:
                self.log_message(f"Unerwarteter Fehler beim Entfernen der Datei: {file_path} - {e}")
                return  # Don't retry for unexpected errors
                # Keine erneuten Versuche bei unerwarteten Fehlern

    def remove_directory_with_retry(self, dir_path, retries=5, delay=1):
        """Entfernt ein Verzeichnis und versucht es bei Bedarf erneut (besonders bei nicht-leeren Verzeichnissen)."""
        dir_path = os.path.abspath(dir_path)
        for i in range(retries):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                self.log_message(f"Verzeichnis entfernt: {dir_path}")
                return
            except OSError as e:
                if e.errno == errno.ENOENT:  # Dir already removed
                    # Verzeichnis bereits entfernt
                    self.log_message(f"Verzeichnis bereits entfernt: {dir_path}")
                    return
                elif e.errno == errno.ENOTEMPTY:  # Directory not empty
                    # Verzeichnis nicht leer
                    self.log_message(f"Verzeichnis nicht leer, versuche Entfernung erneut ({i+1}/{retries}): {dir_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Fehler beim Entfernen des Verzeichnisses: {dir_path} - {e}")
                    time.sleep(delay)  # Wait even for other errors
                    # Auch bei anderen Fehlern warten
            except Exception as e:
                self.log_message(f"Unerwarteter Fehler beim Entfernen des Verzeichnisses: {dir_path} - {e}")
                return

    def setup_ffmpeg(self):
        """Überprüft, ob FFmpeg verfügbar ist."""
        try:
            self.run_subprocess_with_timeout(['ffmpeg', '-version'], 10, "FFmpeg-Prüfung fehlgeschlagen")
            self.ffmpeg_available = True
            self.log_message("FFmpeg erkannt.")
        except FileNotFoundError:
            self.ffmpeg_available = False
            self.log_message("FFmpeg nicht gefunden. Installieren Sie FFmpeg.")
            messagebox.showwarning("FFmpeg fehlt", "FFmpeg ist erforderlich. Bitte installieren Sie es.")
        except Exception as e:
            self.ffmpeg_available = False
            self.log_message(f"FFmpeg-Prüfung fehlgeschlagen: {e}")
            messagebox.showwarning("FFmpeg Fehler", "FFmpeg-Prüfung fehlgeschlagen. Installation überprüfen.")

    def setup_gui(self):
        """Richtet die grafische Benutzeroberfläche ein."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Header ---
        # --- Kopfzeile ---
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        title_label = ttk.Label(header_frame, text="🎬 LinguoAI VideoVoice Pro", font=('Helvetica', 16, 'bold'))
        title_label.pack()

        # --- File Selection ---
        # --- Dateiauswahl ---
        file_frame = ttk.LabelFrame(main_frame, text="Videodateien", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        # Use grid layout for input/output rows
        # Grid-Layout für Eingabe-/Ausgabezeilen verwenden
        source_frame = ttk.Frame(file_frame)
        source_frame.pack(fill=tk.X, pady=2)
        ttk.Label(source_frame, text="📹 Eingabe:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_file, width=40)
        self.source_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(source_frame, text="Durchsuchen", command=self.browse_source, style="Accent.TButton").grid(row=0, column=2, padx=5)

        target_frame = ttk.Frame(file_frame)
        target_frame.pack(fill=tk.X, pady=2)
        ttk.Label(target_frame, text="💾 Ausgabe:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.target_entry = ttk.Entry(target_frame, textvariable=self.target_file, width=40)
        self.target_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(target_frame, text="Durchsuchen", command=self.browse_target, style="Accent.TButton").grid(row=0, column=2, padx=5)

        # Make the entry columns expandable
        # Eingabespalten erweiterbar machen
        source_frame.columnconfigure(1, weight=1)
        target_frame.columnconfigure(1, weight=1)


        # --- Language Selection ---
        # --- Sprachauswahl ---
        lang_frame = ttk.LabelFrame(main_frame, text="Spracheinstellungen", padding="10")
        lang_frame.pack(fill=tk.X, pady=5)

        lang_combo_frame = ttk.Frame(lang_frame)
        lang_combo_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lang_combo_frame, text="🗣️ Zielsprache:").pack(side=tk.LEFT, padx=5)

        # Searchable Combobox
        # Durchsuchbare Combobox
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
        self.lang_combo.set("🇬🇧 Englisch")  # Set default after combobox is created
        # Standardwert festlegen, nachdem die Combobox erstellt wurde
        self.language_search_entry.bind("<Return>", (lambda event: self.lang_combo.focus()))
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_language_selected)

        # --- Merge Options ---
        # --- Zusammenführungsoptionen ---
        options_frame = ttk.LabelFrame(main_frame, text="Zusammenführungsoptionen", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        # Gemini API Key
        # Gemini API-Schlüssel
        brain_frame = ttk.Frame(options_frame)
        brain_frame.pack(fill=tk.X, pady=2)

        # --- Gemini API Key Description ---
        # --- Beschreibung des Gemini API-Schlüssels ---
        gemini_description_label = ttk.Label(
            brain_frame,
            wraplength=600,  # Wrap the text
            # Text umbrechen
            justify=tk.LEFT,  # Left-justify the text
            # Text linksbündig ausrichten
            text="Dieses Programm verwendet die Google Gemini Pro API für optionale Textkürzung.  "
                 "Dies kann helfen, die Gesamtlänge des übersetzten Textes zu reduzieren und gleichzeitig die wichtigsten Informationen zu erhalten.\n"
                 "Ein API-Schlüssel ist erforderlich, um diese Funktion zu nutzen.  Wenn Sie keinen Schlüssel haben, können Sie diesen Schritt überspringen, "
                 "und das Programm wird ohne Kürzung fortfahren."
        )
        gemini_description_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        gemini_link_label = ttk.Label(
            brain_frame,
            text="Hier erhalten Sie einen Gemini API-Schlüssel: ",
            foreground="dodgerblue",  # Changed to a more subtle blue
            # Zu einem subtileren Blau geändert
            cursor="hand2"  # Change cursor on hover
            # Cursor beim Hovern ändern
        )
        gemini_link_label.pack(side=tk.TOP, fill=tk.X)
        gemini_link_label.bind("<Button-1>", lambda e: self.open_webpage("https://makersuite.google.com/app/apikey"))

        self.gemini_key_label = ttk.Label(brain_frame, text="Gemini API-Schlüssel:")
        self.gemini_key_label.pack(side=tk.LEFT, padx=5)
        self.gemini_key_entry = ttk.Entry(brain_frame, textvariable=self.gemini_key, width=30, show="*")
        self.gemini_key_entry.pack(side=tk.LEFT)

        self.start_button = ttk.Button(options_frame, text="🚀 Verarbeitung starten", command=self.start_processing,
                                       style="Success.TButton")
        self.start_button.pack(pady=10)

        # --- Progress ---
        # --- Fortschritt ---
        progress_frame = ttk.LabelFrame(main_frame, text="Fortschritt", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100,
                                            style="Success.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(progress_frame, text="Bereit")
        self.status_label.pack()

        # --- Log ---
        # --- Protokoll ---
        log_frame = ttk.LabelFrame(main_frame, text="Verarbeitungsprotokoll", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, state='disabled', wrap=tk.WORD)  # Enable word wrapping
        # Wortumbruch aktivieren
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def on_language_selected(self, event):
        """Behandelt die Sprachauswahl aus der Combobox."""
        selected_lang_str = self.lang_combo.get()
        for code, (name, flag) in self.languages.items():
            if f"{flag} {name}" == selected_lang_str:
                self.target_language = code  # Update the target language
                # Zielsprache aktualisieren
                self.log_message(f"Ausgewählte Zielsprache: {name} ({code})")
                break

    def update_language_list(self, *args):
        """Filtert die Sprachliste basierend auf der Sucheingabe."""
        search_term = self.language_search_var.get().lower()
        filtered_languages = [
            f"{flag} {name}"
            for code, (name, flag) in self.languages.items()
            if search_term in name.lower() or search_term in code.lower() or search_term in flag.lower()
        ]
        self.lang_combo['values'] = filtered_languages
        if filtered_languages:
            self.lang_combo.set(filtered_languages[0])  # Set to the first match
            # Auf erste Übereinstimmung setzen
        else:
            self.lang_combo.set('')  # Clear if no matches
            # Leeren, wenn keine Übereinstimmungen
    def browse_source(self):
        """Öffnet einen Dateiauswahldialog zum Auswählen der Quellvideodatei."""
        filename = filedialog.askopenfilename(
            title="Videodatei auswählen",
            filetypes=[("Videodateien", "*.mp4 *.avi *.mkv *.mov"), ("Alle Dateien", "*.*")]
        )
        if filename:
            self.source_file.set(filename)
            base = os.path.splitext(filename)[0]
            self.target_file.set(f"{base}_übersetzt.mp4")
            self.source_entry.xview_moveto(1) #Scroll to the end
            # Zum Ende scrollen
            self.target_entry.xview_moveto(1)

    def browse_target(self):
        """Öffnet einen Dateiauswahldialog zum Auswählen des Zielvideopfads."""
        filename = filedialog.asksaveasfilename(
            title="Übersetztes Video speichern",
            defaultextension=".mp4",
            filetypes=[("MP4-Dateien", "*.mp4")]
        )
        if filename:
            self.target_file.set(filename)
            self.source_entry.xview_moveto(1)  # Scroll to the end
            # Zum Ende scrollen
            self.target_entry.xview_moveto(1)
    def log_message(self, message):
        """Protokolliert eine Nachricht in der GUI und der internen Protokollliste."""
        self.log_messages.append(message)
        self.log_text.config(state='normal')  # Temporarily enable
        # Temporär aktivieren
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')  # Disable again
        # Wieder deaktivieren
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def load_api_keys_from_environment(self):
        """Lädt API-Schlüssel aus Umgebungsvariablen."""
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_key.set(self.gemini_api_key)  # Load into GUI
        # In GUI laden

    def start_processing(self):
        """Startet die Videoverarbeitung in einem separaten Thread."""
        if not self.ffmpeg_available:
            messagebox.showerror("Fehler", "FFmpeg ist erforderlich!")
            return
        if not self.source_file.get() or not self.target_file.get():
            messagebox.showerror("Fehler", "Eingabe- und Ausgabedateien auswählen.")
            return

        try:
            lang_str = self.lang_combo.get()
            self.target_language = next(
                (code for code, (name, flag) in self.languages.items() if f"{flag} {name}" == lang_str), None
            )
            if self.target_language is None:
                raise ValueError("Ungültige Sprache ausgewählt.")
        except ValueError as e:
            messagebox.showerror("Fehler", str(e))
            return

        self.gemini_api_key = self.gemini_key.get()
        if self.gemini_api_key:
            self.init_gemini()
        else:
            self.log_message("Gemini API-Schlüssel nicht angegeben. Überspringe Kürzung.")

        self.start_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.log_text.config(state='normal')
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state='disabled')
        self.log_messages = []
        threading.Thread(target=self.process_video, daemon=True).start()

    def load_whisper_model(self):
        """Lädt das Faster Whisper Modell."""
        try:
            self.log_message(f"Lade Whisper Modell ({self.model_size}) auf {self.device}...")
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.log_message("Whisper Modell erfolgreich geladen.")
        except Exception as e:
            self.log_message(f"Fehler beim Laden des Whisper Modells: {e}")
            raise

    def extract_audio_in_chunks(self, video_path: str) -> Tuple[List[str], float]:
        """Extrahiert Audio aus dem Video in Chunks."""
        self.log_message(f"Extrahiere Audio aus: {video_path}")
        audio_chunks = []
        try:
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Fehler beim Abrufen der Videodauer")
            total_duration = float(duration_output.strip())
            self.log_message(f"Gesamte Videodauer: {total_duration:.2f} Sekunden")

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
            self.log_message(f"Fehler während der Audio-Extraktion: {e}")
            raise  # Re-raise the exception to be handled in process_video
            # Ausnahme erneut auslösen, um sie in process_video zu behandeln

    def transcribe_audio_in_chunks(self, audio_chunks: List[str], language: str) -> Tuple[List[Dict], int]:
        """Transkribiert Audio-Chunks mit Faster Whisper."""
        all_segments = []
        total_words = 0
        if self.whisper_model is None:
            raise ValueError("Whisper Modell nicht geladen.")

        try:
            for i, chunk_path in enumerate(audio_chunks):
                self.log_message(f"Transkribiere Chunk {i+1}/{len(audio_chunks)}: {chunk_path}")
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
                self.log_message(f"Transkription von Chunk {i+1} abgeschlossen.")
                self.root.update()  # Update GUI
                # GUI aktualisieren

                self.remove_file_with_retry(chunk_path)
                self.remove_directory_with_retry(os.path.dirname(chunk_path))

            self.log_message(f"Gesamte transkribierte Wörter: {total_words}")
            return all_segments, total_words
        except Exception as e:
            self.log_message(f"Fehler während der Transkription: {e}")
            raise

    def translate_and_refine_segments(self, segments: List[Dict]) -> Tuple[List[Dict], int]:
        """Übersetzt Segmente und behandelt potenzielle Übersetzungsfehler."""
        translated_segments = []
        total_words_translated = 0
        translator = GoogleTranslator(source='auto', target=self.target_language)

        try:
            for segment in segments:
                self.log_message(f"Übersetze Segment: '{segment['text'][:50]}...' nach {self.target_language}")
                translated_text = translator.translate(segment['text'])
                if not isinstance(translated_text, str):
                    self.log_message(
                        f"WARNUNG: translator.translate hat keinen String zurückgegeben. Typ: {type(translated_text)}, Wert: {translated_text}")
                    translated_text = ""  # Set to empty string
                    # Auf leeren String setzen
                translated_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': translated_text
                })
                total_words_translated += len(translated_text.split())
                self.log_message(f"Übersetztes Segment: '{translated_text[:50]}...'")
            return translated_segments, total_words_translated
        except Exception as e:
            self.log_message(f"Fehler während der Übersetzung: {e}")
            raise

    def generate_tts_audio_for_segments(self, translated_segments: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
        """Generiert TTS-Audio für einen Batch übersetzter Segmente."""
        tts_chunks = []
        updated_segments = []
        total_duration = 0
        try:
            for i, segment in enumerate(translated_segments):
                text = segment['text']
                self.log_message(f"Generiere TTS für Segment {i+1}/{len(translated_segments)}: '{text[:50]}...'")
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
                                                                        "Fehler beim Abrufen der TTS-Segmentdauer")
                    segment_duration = float(duration_output.strip())
                    updated_segments.append({
                        'start': total_duration,
                        'end': total_duration + segment_duration,
                        'text': text,
                    })
                    total_duration += segment_duration
                except Exception as e:
                    self.log_message(f"Fehler beim Generieren von TTS für Segment {i + 1}: {e}")
                    # Clean up any created files in this batch so far
                    # Alle bisher erstellten Dateien in diesem Batch aufräumen
                    for file in tts_chunks:
                        self.remove_file_with_retry(file)
                        self.remove_directory_with_retry(os.path.dirname(file))  # Try to remove dir
                        # Versuchen, Verzeichnis zu entfernen
                    return None, []  # Indicate failure
                    # Fehler anzeigen

                self.log_message(f"TTS für Segment {i+1} generiert.")
                self.root.update()  # Keep GUI responsive
                # GUI reaktionsfähig halten

            self.log_message(f"Führe {len(tts_chunks)} TTS-Chunks zusammen...")
            if tts_chunks:
                merged_audio_path = os.path.join(self.process_temp_dir, f"merged_tts_audio_{int(time.time())}.mp3")
                self.merge_audio_chunks(tts_chunks, merged_audio_path)
                self.log_message(f"Zusammengeführtes TTS-Audio: {merged_audio_path}")

                for tts_file in tts_chunks:
                    self.remove_file_with_retry(tts_file)
                    self.remove_directory_with_retry(os.path.dirname(tts_file))

                return merged_audio_path, updated_segments
            else:
                self.log_message("Keine TTS-Segmente generiert.")
                return None, []

        except Exception as e:
            self.log_message(f"Fehler während der TTS-Generierung/Zusammenführung: {e}")
            for file in tts_chunks:  # Corrected
                # Korrigiert
                self.remove_file_with_retry(file)
                self.remove_directory_with_retry(os.path.dirname(file))  # and dir
                # und Verzeichnis
            return None, []

    def open_webpage(self, url):
        """Öffnet eine Webseite im Standardbrowser."""
        import webbrowser
        webbrowser.open(url)

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        print("Lade NLTK punkt Tokenizer herunter...")
        nltk.download('punkt')
    app = LinguoAIVideoVoicePro()
    app.root.mainloop()