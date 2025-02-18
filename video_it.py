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
        # --- Parametri di configurazione ---
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
        self.hw_accel_info = self.detect_hardware_acceleration()  # Salva le informazioni
        self.hw_accel = self.hw_accel_info['accel'] if self.hw_accel_info else None

        # --- Stato interno ---
        self.whisper_model: Optional[WhisperModel] = None
        self.current_process: Optional[subprocess.Popen] = None
        self.process_start_time: Optional[float] = None
        self.process_temp_dir: str = ""

        # --- Componenti GUI ---
        self.root = ttk.Window(themename="darkly")
        self.root.title("LinguoAI VideoVoice Pro")
        self.root.geometry("640x850")  # Imposta la dimensione iniziale

        self.source_file = tk.StringVar()
        self.target_file = tk.StringVar()
        self.selected_language = tk.StringVar(value="en")
        self.gemini_key = tk.StringVar()  # Chiave API Gemini
        self.progress_var = tk.DoubleVar(value=0)
        self.log_messages: List[str] = []

        self.languages = {
            "af": ("Afrikaans", "🇿🇦"),
            "sq": ("Albanisch", "🇦🇱"),
            "am": ("Amharisch", "🇪🇹"),
            "ar": ("Arabo", "🇸🇦"),
            "eu": ("Basco", "🇪🇸"),
            "bn": ("Bengalese", "🇧🇩"),
            "bs": ("Bosniaco", "🇧🇦"),
            "bg": ("Bulgaro", "🇧🇬"),
            "ca": ("Catalano", "🇦🇩"),
            "zh-CN": ("Cinese (Semplificato)", "🇨🇳"),
            "zh-TW": ("Cinese (Tradizionale)", "🇹🇼"),
            "hr": ("Croato", "🇭🇷"),
            "cs": ("Ceco", "🇨🇿"),
            "da": ("Danese", "🇩🇰"),
            "nl": ("Olandese", "🇳🇱"),
            "en": ("Inglese", "🇬🇧"),
            "et": ("Estone", "🇪🇪"),
            "tl": ("Filippino", "🇵🇭"),
            "fi": ("Finlandese", "🇫🇮"),
            "fr": ("Francese", "🇫🇷"),
            "gl": ("Galiziano", "🇪🇸"),
            "de": ("Tedesco", "🇩🇪"),
            "el": ("Greco", "🇬🇷"),
            "gu": ("Gujarati", "🇮🇳"),
            "ha": ("Hausa", "🇳🇬"),
            "he": ("Ebraico", "🇮🇱"),
            "hi": ("Hindi", "🇮🇳"),
            "hu": ("Ungherese", "🇭🇺"),
            "is": ("Islandese", "🇮🇸"),
            "id": ("Indonesiano", "🇮🇩"),
            "it": ("Italiano", "🇮🇹"),
            "ja": ("Giapponese", "🇯🇵"),
            "jw": ("Giavanese", "🇮🇩"),
            "kn": ("Kannada", "🇮🇳"),
            "km": ("Khmer", "🇰🇭"),
            "ko": ("Coreano", "🇰🇷"),
            "la": ("Latino", "🇻🇦"),
            "lv": ("Lettone", "🇱🇻"),
            "lt": ("Lituano", "🇱🇹"),
            "ms": ("Malese", "🇲🇾"),
            "mr": ("Marathi", "🇮🇳"),
            "ml": ("Malayalam", "🇮🇳"),
            "my": ("Myanmar (Birmano)", "🇲🇲"),
            "ne": ("Nepalese", "🇳🇵"),
            "no": ("Norvegese", "🇳🇴"),
            "pa": ("Punjabi", "🇮🇳"),
            "pl": ("Polacco", "🇵🇱"),
            "pt": ("Portoghese", "🇵🇹"),
            "ro": ("Rumeno", "🇷🇴"),
            "ru": ("Russo", "🇷🇺"),
            "sr": ("Serbo", "🇷🇸"),
            "si": ("Singalese", "🇱🇰"),
            "sk": ("Slovacco", "🇸🇰"),
            "sl": ("Sloveno", "🇸🇮"),
            "es": ("Spagnolo", "🇪🇸"),
            "su": ("Sundanese", "🇮🇩"),
            "sw": ("Swahili", "🇰🇪"),
            "sv": ("Svedese", "🇸🇪"),
            "ta": ("Tamil", "🇮🇳"),
            "te": ("Telugu", "🇮🇳"),
            "th": ("Thai", "🇹🇭"),
            "tr": ("Turco", "🇹🇷"),
            "uk": ("Ucraino", "🇺🇦"),
            "ur": ("Urdu", "🇵🇰"),
            "vi": ("Vietnamita", "🇻🇳"),
            "cy": ("Gallese", "🇬🇧")
        }

        # --- Inizializzazione ---
        self.setup_gui()
        self.setup_ffmpeg()
        self.load_api_keys_from_environment()
        self.load_whisper_model()
        if self.gemini_api_key:
            self.init_gemini()
        self.log_hardware_acceleration()

    def seconds_to_srt_time(self, seconds: float) -> str:
        """Converte i secondi nel formato orario SRT (HH:MM:SS,mmm)."""
        milliseconds = int((seconds * 1000) % 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    def create_srt_file(self, segments: List[Dict], output_path: str):
        """Crea un file SRT dai segmenti trascritti/tradotti."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments):
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text']

                    # Converti i secondi nel formato orario SRT
                    start_time_srt = self.seconds_to_srt_time(start_time)
                    end_time_srt = self.seconds_to_srt_time(end_time)

                    f.write(f"{i + 1}\n")
                    f.write(f"{start_time_srt} --> {end_time_srt}\n")
                    f.write(f"{text}\n\n")

            self.log_message(f"File SRT creato: {output_path}")

        except Exception as e:
            self.log_message(f"Errore durante la creazione del file SRT: {e}")

    def log_hardware_acceleration(self):
        """Registra le informazioni sull'accelerazione hardware."""
        if self.hw_accel_info:
            self.log_message(f"Accelerazione hardware rilevata: {self.hw_accel_info['accel']} ({self.hw_accel_info['info']})")
        else:
            self.log_message("Nessuna accelerazione hardware rilevata.")

    def detect_hardware_acceleration(self):
        """Rileva l'accelerazione hardware (NVIDIA, Intel, AMD)."""
        try:
            # NVIDIA
            try:
                subprocess.run(['nvidia-smi'], check=True, capture_output=True)
                return {'accel': 'cuda', 'info': 'GPU NVIDIA rilevata'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # Intel (Quick Sync)
            try:
                output = subprocess.run(['vainfo'], check=True, capture_output=True, text=True).stdout
                if "VA-API" in output:  # Controllo molto approssimativo, può essere migliorato
                    return {'accel': 'qsv', 'info': 'Intel Quick Sync rilevato'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # AMD (AMF)  - un po' più complesso, poiché non esiste un semplice "amfinfo"
            #  Si potrebbe cercare driver/dispositivi specifici, ma questo è specifico per il sistema operativo.
            #  Qui un controllo molto semplice e incompleto per Linux:
            if os.name == 'posix':  # Linux/macOS
                try:
                    output = subprocess.run(['lspci', '-v'], check=True, capture_output=True, text=True).stdout
                    if "Advanced Micro Devices, Inc. [AMD/ATI]" in output:
                        return {'accel': 'h264_vaapi', 'info': 'GPU AMD rilevata (VAAPI)'}  # Presunzione!
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass

            return None  # Nessuna accelerazione hardware trovata

        except Exception as e:
            self.log_message(f"Errore durante il rilevamento dell'accelerazione hardware: {e}")
            return None
    def init_gemini(self):
        """Inizializza il modello Gemini Pro."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.log_message("Modello Gemini Pro inizializzato.")
        except Exception as e:
            self.log_message(f"Errore durante l'inizializzazione di Gemini Pro: {e}")
            messagebox.showerror("Errore Gemini", f"Impossibile inizializzare Gemini Pro: {e}")
            self.model = None  # Imposta a None in caso di errore

    def check_process_timeout(self):
        """Verifica se il processo complessivo ha superato il tempo massimo consentito."""
        if self.process_start_time and time.time() - self.process_start_time > self.process_timeout:
            if self.current_process:
                self.log_message(f"Timeout del processo! Terminazione del processo (PID: {self.current_process.pid})...")
                self.kill_process_tree(self.current_process.pid)  # Termina l'albero dei processi!
            raise TimeoutError("Il processo ha superato il tempo massimo consentito")

    def kill_process_tree(self, pid):
        """Termina un processo e tutti i suoi processi figlio."""
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):  # Ottieni tutti i figli/nipoti
                self.log_message(f"Terminazione del processo figlio: {child.pid}")
                child.kill()
            self.log_message(f"Terminazione del processo padre: {parent.pid}")
            parent.kill()
        except psutil.NoSuchProcess:
            self.log_message(f"Processo con PID {pid} non trovato.")
        except Exception as e:
            self.log_message(f"Errore durante la terminazione dell'albero dei processi: {e}")

    def run_subprocess_with_timeout(self, command, timeout, error_message):
        """Esegue un sottoprocesso con timeout dinamico e controlli di attività."""
        try:
            self.log_message(f"Esecuzione del comando con timeout {timeout}: {' '.join(command)}")
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.log_message(f"Processo avviato con PID: {self.current_process.pid}")

            start_time = time.time()
            last_output_time = start_time

            stdout, stderr = self.current_process.communicate(timeout=timeout)  # Usa communicate!
            retcode = self.current_process.returncode

            if retcode != 0:
                self.log_message(f"Processo fallito con codice di errore {retcode}:")
                self.log_message(f"Stdout: {stdout}")
                self.log_message(f"Stderr: {stderr}")
                raise subprocess.CalledProcessError(retcode, command, stdout, stderr)

            self.log_message("Comando completato con successo.")
            return stdout, stderr

        except subprocess.TimeoutExpired:
            self.log_message(f"Timeout del processo dopo {timeout} secondi")
            self.kill_process_tree(self.current_process.pid)  # Termina l'albero dei processi!
            stdout, stderr = self.current_process.communicate()  # Ottieni qualsiasi output rimanente
            self.log_message(f"Stdout: {stdout}")
            self.log_message(f"Stderr: {stderr}")
            raise TimeoutError(error_message)

        except Exception as e:
            self.log_message(f"Si è verificato un errore inatteso: {e}")
            if self.current_process:
                self.kill_process_tree(self.current_process.pid)  # Termina se ancora in esecuzione
            raise
        finally:
            self.current_process = None

    def extract_audio_chunk(self, video_path, audio_path, start_time, duration):
        """Estrae una porzione di audio dal video."""
        command = [
            "ffmpeg",
            "-y",  # Sovrascrivi i file di output senza chiedere
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-vn",  # Disabilita video
            "-acodec", "libmp3lame",
            "-q:a", "2",  # MP3 di buona qualità
            "-loglevel", "error",  # Riduci la verbosità
            audio_path
        ]
        if self.hw_accel:
            command.insert(1, "-hwaccel")
            command.insert(2, self.hw_accel)

        try:
            self.run_subprocess_with_timeout(command, self.chunk_timeout, "Estrazione audio scaduta per timeout")
            self.log_message(f"Porzione audio estratta: '{audio_path}'")
        except Exception as e:
            self.log_message(f"Errore durante l'estrazione della porzione audio: {e}")
            raise

    def batch_segments(self, segments: List[Dict]) -> List[List[Dict]]:
        """Divide i segmenti in batch più piccoli per TTS."""
        batched_segments = []
        for i in range(0, len(segments), self.tts_batch_size):
            batch = segments[i:i + self.tts_batch_size]
            batched_segments.append(batch)
        return batched_segments

    def validate_audio_chunk(self, chunk_path: str) -> bool:
        """Valida una porzione audio usando ffprobe."""
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
                f"Validazione della porzione audio scaduta per timeout: {chunk_path}"
            )
            if stdout.strip():
                self.log_message(f"Porzione audio validata: {chunk_path}")
                return True
            else:
                self.log_message(f"Validazione della porzione audio fallita (nessuna durata): {chunk_path}")
                return False
        except Exception as e:
            self.log_message(f"Errore durante la validazione della porzione audio {chunk_path}: {e}")
            return False

    def merge_audio_chunks(self, audio_chunks, output_path):
        """Unisce più porzioni audio in un singolo file usando FFmpeg."""
        if not audio_chunks:
            self.log_message("Nessuna porzione audio da unire.")
            return

        valid_chunks = [chunk for chunk in audio_chunks if self.validate_audio_chunk(chunk)]
        if not valid_chunks:
            self.log_message("Nessuna porzione audio valida da unire.")
            return

        timestamp = int(time.time())
        temp_dir = os.path.join(tempfile.gettempdir(), f"audio_merge_temp_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)

        list_file_path = os.path.join(temp_dir, "chunk_list.txt")
        try:
            with open(list_file_path, "w") as f:
                for chunk_path in valid_chunks:
                    abs_chunk_path = os.path.abspath(chunk_path)  # Usa il percorso assoluto
                    f.write(f"file '{abs_chunk_path}'\n")

            command = [
                "ffmpeg",
                "-y",  # Sovrascrivi
                "-f", "concat",
                "-safe", "0",  # Richiesto per i percorsi assoluti con concat
                "-i", list_file_path,
                "-c", "copy",
                "-loglevel", "error",
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            # Calcola un timeout dinamico in base al numero di porzioni.
            merge_timeout = len(valid_chunks) * 10 + 30  # 10 secondi per porzione + 30 base
            self.run_subprocess_with_timeout(command, merge_timeout, "Unione audio scaduta per timeout")
            self.log_message(f"Porzioni audio unite: '{output_path}'")

        except Exception as e:
            self.log_message(f"Errore durante l'unione delle porzioni audio: {e}")
            raise
        finally:
            self.remove_file_with_retry(list_file_path)  # Usa la funzione di riprova
            self.remove_directory_with_retry(temp_dir)  # e per la directory

    def merge_video_audio(self, audio_file):
        """Unisce l'audio finale con il video originale."""
        try:
            output_path = self.target_file.get()
            video_path = self.source_file.get()

            # Ottieni la durata del video usando ffprobe
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Impossibile ottenere la durata del video")
            total_duration = float(duration_output.strip())
            self.log_message(f"Durata video per l'unione: {total_duration:.2f} secondi")

            # Calcola il timeout dinamico (es., 3x durata + 120 secondi)
            dynamic_timeout = int(3 * total_duration + 120)
            self.log_message(f"Timeout dinamico per l'unione: {dynamic_timeout} secondi")

            command = [
                'ffmpeg',
                '-y',  # Sovrascrivi il file di output
                '-i', video_path,
                '-i', audio_file,
                '-c:v', 'h264_nvenc' if self.hw_accel == 'cuda' else 'libx264',  # H.265
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',  # Termina la codifica quando il flusso più corto termina
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            self.run_subprocess_with_timeout(command, dynamic_timeout, "Unione di video e audio scaduta per timeout")
            self.log_message(f"Video e audio uniti: {output_path}")
        except Exception as e:
            self.log_message(f"Errore durante l'unione di video e audio: {e}")
            raise

    def shorten_text_with_gemini(self, text: str) -> str:
        """Accorcia il testo usando Gemini Pro, gestendo errori e limiti di frequenza."""
        if self.model is None:
            self.log_message("Modello Gemini Pro non inizializzato. Salto l'accorciamento.")
            return text

        try:
            prompt = f"Per favore, accorcia il seguente testo preservando le informazioni chiave:\n\n{text}"
            time.sleep(1.5)  # Limite di frequenza: Pausa di 1.5 secondi
            response = self.model.generate_content(prompt)
            if response and response.text:
                shortened_text = response.text
                self.log_message("Testo accorciato con Gemini Pro.")
                return shortened_text
            else:
                self.log_message("Gemini Pro ha restituito una risposta vuota. Utilizzo il testo originale.")
                return text
        except Exception as e:
            self.log_message(f"Errore durante l'accorciamento del testo con Gemini: {e}")
            return text

    def process_video(self):
        """Flusso di lavoro principale per l'elaborazione video."""
        self.process_start_time = time.time()
        timestamp = int(time.time())
        self.process_temp_dir = os.path.join(tempfile.gettempdir(), f"process_temp_{timestamp}")
        os.makedirs(self.process_temp_dir, exist_ok=True)

        try:
            self.log_message("Avvio elaborazione video...")
            self.progress_var.set(5)

            # Imposta il controllo del timeout *prima* di avviare qualsiasi attività.
            def check_timeout():
                self.check_process_timeout()
                self.root.after(1000, check_timeout)  # Controlla ogni secondo

            check_timeout()  # Avvia il controllo del timeout

            self.progress_var.set(10)
            audio_chunks, total_duration = self.extract_audio_in_chunks(self.source_file.get())
            self.progress_var.set(25)

            segments, total_words_original = self.transcribe_audio_in_chunks(audio_chunks, self.target_language)
            if not segments:
                raise Exception("Trascrizione fallita.")
            self.progress_var.set(45)

            translated_segments, total_words_translated = self.translate_and_refine_segments(segments)

            # --- Accorciamento del testo con Gemini (Opzionale) ---
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
                    self.root.update()  # Aggiorna GUI
                translated_segments = shortened_segments

                shortening_change = (
                    ((total_words_translated - total_words_shortened) / total_words_translated) * 100
                    if total_words_translated else 0
                )
                self.log_message(f"L'accorciamento ha ridotto il conteggio delle parole del: {shortening_change:.2f}%")
            else:
                total_words_shortened = total_words_translated
                self.log_message("Chiave API Gemini non fornita. Salto l'accorciamento.")

            self.progress_var.set(60)

            # Creazione del file SRT (Esempio)
            srt_output_path = os.path.splitext(self.target_file.get())[0] + ".srt"  # Corrisponde al nome del video di output
            self.create_srt_file(translated_segments, srt_output_path)

            # --- Rapporto sul conteggio delle parole ---
            if total_words_original > 0:
                translation_change = (total_words_translated / total_words_original) * 100
                self.log_message(f"Conteggio parole originale: {total_words_original}")
                self.log_message(f"Conteggio parole tradotto: {total_words_translated}")
                self.log_message(f"Variazione del conteggio delle parole tradotte: {translation_change:.2f}%")
            else:
                self.log_message("Il conteggio delle parole originale è zero. Salto la percentuale.")

            # --- TTS e unione audio ---
            batched_translated_segments = self.batch_segments(translated_segments)
            all_updated_segments = []
            merged_audio_files = []

            for i, batch in enumerate(batched_translated_segments):
                self.log_message(f"Elaborazione batch TTS {i+1}/{len(batched_translated_segments)}")
                merged_audio_path, updated_segments = self.generate_tts_audio_for_segments(batch)
                if merged_audio_path:
                    all_updated_segments.extend(updated_segments)
                    merged_audio_files.append(merged_audio_path)
                else:
                    self.log_message(f"Batch TTS {i+1} fallito.")
                    # Pulisci *tutti* i file TTS creati in precedenza in caso di errore
                    for file_path in merged_audio_files:
                        self.remove_file_with_retry(file_path)
                    raise Exception(f"Generazione audio TTS fallita per il batch {i + 1}.")

            # --- Unione audio finale (se più batch) ---
            if len(merged_audio_files) > 1:
                final_merged_audio_path = os.path.join(self.process_temp_dir,
                                                        f"final_merged_tts_audio_{timestamp}.mp3")
                self.merge_audio_chunks(merged_audio_files, final_merged_audio_path)
                self.log_message(f"Audio TTS unito finale: {final_merged_audio_path}")
            elif merged_audio_files:
                final_merged_audio_path = merged_audio_files[0]
                self.log_message(f"Audio TTS unito finale (batch singolo): {final_merged_audio_path}")
            else:
                raise Exception("Nessun audio TTS generato.")

            # --- Pulisci i file TTS intermedi ---
            for file_path in merged_audio_files:
                if file_path != final_merged_audio_path:  # Non eliminare il file finale!
                    self.remove_file_with_retry(file_path)

            self.progress_var.set(80)

            # --- Unione finale Video/Audio ---
            self.merge_video_audio(final_merged_audio_path)
            self.progress_var.set(95)

            self.log_message("Elaborazione completata con successo! 🎉")
            messagebox.showinfo("Successo", "Elaborazione video completata!")


        except TimeoutError as e:
            self.log_message(f"Elaborazione scaduta per timeout: {str(e)}")
            messagebox.showerror("Errore", f"Elaborazione scaduta per timeout: {str(e)}")
        except Exception as e:
            self.log_message(f"Errore: {str(e)}")
            messagebox.showerror("Errore", f"Elaborazione fallita: {str(e)}")
        finally:
            # --- Pulizia ---
            self.progress_var.set(0)
            self.process_start_time = None
            self.current_process = None
            self.start_button.config(state=tk.NORMAL)  # Riabilita il pulsante
            self.remove_directory_with_retry(self.process_temp_dir)

    def remove_file_with_retry(self, file_path, retries=3, delay=0.5):
        """Rimuove un file, riprovando se necessario."""
        file_path = os.path.abspath(file_path)  # Usa il percorso assoluto
        for i in range(retries):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                self.log_message(f"File rimosso: {file_path}")
                return  # Successo
            except OSError as e:
                if e.errno == errno.ENOENT:  # File non trovato - già eliminato
                    self.log_message(f"File non trovato (già rimosso): {file_path}")
                    return
                if i < retries - 1:  # Non aspettare l'ultimo tentativo
                    self.log_message(f"Riprovo a rimuovere il file ({i+1}/{retries}): {file_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Errore durante la rimozione del file dopo molteplici tentativi: {file_path} - {e}")
            except Exception as e:
                self.log_message(f"Errore inatteso durante la rimozione del file: {file_path} - {e}")
                return  # Non riprovare per errori inattesi

    def remove_directory_with_retry(self, dir_path, retries=5, delay=1):
        """Rimuove una directory, riprovando se necessario (specialmente per non vuote)."""
        dir_path = os.path.abspath(dir_path)
        for i in range(retries):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                self.log_message(f"Directory rimossa: {dir_path}")
                return
            except OSError as e:
                if e.errno == errno.ENOENT:  # Directory già rimossa
                    self.log_message(f"Directory già rimossa: {dir_path}")
                    return
                elif e.errno == errno.ENOTEMPTY:  # Directory non vuota
                    self.log_message(f"Directory non vuota, riprovo a rimuovere ({i+1}/{retries}): {dir_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Errore durante la rimozione della directory: {dir_path} - {e}")
                    time.sleep(delay)  # Aspetta anche per altri errori
            except Exception as e:
                self.log_message(f"Errore inatteso durante la rimozione della directory: {dir_path} - {e}")
                return

    def setup_ffmpeg(self):
        """Verifica se FFmpeg è disponibile."""
        try:
            self.run_subprocess_with_timeout(['ffmpeg', '-version'], 10, "Controllo FFmpeg fallito")
            self.ffmpeg_available = True
            self.log_message("FFmpeg rilevato.")
        except FileNotFoundError:
            self.ffmpeg_available = False
            self.log_message("FFmpeg non trovato. Installa FFmpeg.")
            messagebox.showwarning("FFmpeg Mancante", "FFmpeg è richiesto. Per favore, installalo.")
        except Exception as e:
            self.ffmpeg_available = False
            self.log_message(f"Controllo FFmpeg fallito: {e}")
            messagebox.showwarning("Errore FFmpeg", "Controllo FFmpeg fallito. Controlla l'installazione.")

    def setup_gui(self):
        """Imposta l'interfaccia utente grafica."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Intestazione ---
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        title_label = ttk.Label(header_frame, text="🎬 LinguoAI VideoVoice Pro", font=('Helvetica', 16, 'bold'))
        title_label.pack()

        # --- Selezione File ---
        file_frame = ttk.LabelFrame(main_frame, text="File Video", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        # Usa il layout a griglia per le righe di input/output
        source_frame = ttk.Frame(file_frame)
        source_frame.pack(fill=tk.X, pady=2)
        ttk.Label(source_frame, text="📹 Input:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_file, width=40)
        self.source_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(source_frame, text="Sfoglia", command=self.browse_source, style="Accent.TButton").grid(row=0, column=2, padx=5)

        target_frame = ttk.Frame(file_frame)
        target_frame.pack(fill=tk.X, pady=2)
        ttk.Label(target_frame, text="💾 Output:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.target_entry = ttk.Entry(target_frame, textvariable=self.target_file, width=40)
        self.target_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(target_frame, text="Sfoglia", command=self.browse_target, style="Accent.TButton").grid(row=0, column=2, padx=5)

        # Rendi espandibili le colonne di input
        source_frame.columnconfigure(1, weight=1)
        target_frame.columnconfigure(1, weight=1)


        # --- Selezione Lingua ---
        lang_frame = ttk.LabelFrame(main_frame, text="Impostazioni Voce", padding="10")
        lang_frame.pack(fill=tk.X, pady=5)

        lang_combo_frame = ttk.Frame(lang_frame)
        lang_combo_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lang_combo_frame, text="🗣️ Lingua di destinazione:").pack(side=tk.LEFT, padx=5)

        # Combobox ricercabile
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
        self.lang_combo.set("🇬🇧 English")  # Imposta il valore predefinito dopo che la combobox è stata creata
        self.language_search_entry.bind("<Return>", (lambda event: self.lang_combo.focus()))
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_language_selected)

        # --- Opzioni di Unione ---
        options_frame = ttk.LabelFrame(main_frame, text="Opzioni di Unione", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        # Chiave API Gemini
        brain_frame = ttk.Frame(options_frame)
        brain_frame.pack(fill=tk.X, pady=2)

        # --- Descrizione Chiave API Gemini ---
        gemini_description_label = ttk.Label(
            brain_frame,
            wraplength=600,  # A capo il testo
            justify=tk.LEFT,  # Giustifica il testo a sinistra
            text="Questo programma utilizza l'API Google Gemini Pro per l'accorciamento opzionale del testo.  "
                 "Questo può aiutare a ridurre la lunghezza complessiva del testo tradotto preservando le informazioni chiave.\n"
                 "È richiesta una chiave API per utilizzare questa funzionalità. Se non hai una chiave, puoi saltare questo passaggio, "
                 "e il programma procederà senza accorciare."
        )
        gemini_description_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        gemini_link_label = ttk.Label(
            brain_frame,
            text="Ottieni una chiave API Gemini qui: ",
            foreground="dodgerblue",  # Cambiato in un blu più tenue
            cursor="hand2"  # Cambia il cursore al passaggio del mouse
        )
        gemini_link_label.pack(side=tk.TOP, fill=tk.X)
        gemini_link_label.bind("<Button-1>", lambda e: self.open_webpage("https://makersuite.google.com/app/apikey"))

        self.gemini_key_label = ttk.Label(brain_frame, text="Chiave API Gemini:")
        self.gemini_key_label.pack(side=tk.LEFT, padx=5)
        self.gemini_key_entry = ttk.Entry(brain_frame, textvariable=self.gemini_key, width=30, show="*")
        self.gemini_key_entry.pack(side=tk.LEFT)

        self.start_button = ttk.Button(options_frame, text="🚀 Avvia Elaborazione", command=self.start_processing,
                                       style="Success.TButton")
        self.start_button.pack(pady=10)

        # --- Progresso ---
        progress_frame = ttk.LabelFrame(main_frame, text="Progresso", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100,
                                            style="Success.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(progress_frame, text="Pronto")
        self.status_label.pack()

        # --- Log ---
        log_frame = ttk.LabelFrame(main_frame, text="Log di Elaborazione", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, state='disabled', wrap=tk.WORD)  # Abilita il ritorno a capo delle parole
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def on_language_selected(self, event):
        """Gestisce la selezione della lingua dalla combobox."""
        selected_lang_str = self.lang_combo.get()
        for code, (name, flag) in self.languages.items():
            if f"{flag} {name}" == selected_lang_str:
                self.target_language = code  # Aggiorna la lingua di destinazione
                self.log_message(f"Lingua di destinazione selezionata: {name} ({code})")
                break

    def update_language_list(self, *args):
        """Filtra l'elenco delle lingue in base all'input di ricerca."""
        search_term = self.language_search_var.get().lower()
        filtered_languages = [
            f"{flag} {name}"
            for code, (name, flag) in self.languages.items()
            if search_term in name.lower() or search_term in code.lower() or search_term in flag.lower()
        ]
        self.lang_combo['values'] = filtered_languages
        if filtered_languages:
            self.lang_combo.set(filtered_languages[0])  # Imposta sulla prima corrispondenza
        else:
            self.lang_combo.set('')  # Pulisci se non ci sono corrispondenze
    def browse_source(self):
        """Apre una finestra di dialogo per selezionare il video sorgente."""
        filename = filedialog.askopenfilename(
            title="Seleziona File Video",
            filetypes=[("File video", "*.mp4 *.avi *.mkv *.mov"), ("Tutti i file", "*.*")]
        )
        if filename:
            self.source_file.set(filename)
            base = os.path.splitext(filename)[0]
            self.target_file.set(f"{base}_tradotto.mp4")
            self.source_entry.xview_moveto(1) #Scorri fino alla fine
            self.target_entry.xview_moveto(1)

    def browse_target(self):
        """Apre una finestra di dialogo per selezionare il percorso del video di destinazione."""
        filename = filedialog.asksaveasfilename(
            title="Salva Video Tradotto",
            defaultextension=".mp4",
            filetypes=[("File MP4", "*.mp4")]
        )
        if filename:
            self.target_file.set(filename)
            self.source_entry.xview_moveto(1)  # Scorri fino alla fine
            self.target_entry.xview_moveto(1)
    def log_message(self, message):
        """Registra un messaggio nella GUI e nella lista di log interna."""
        self.log_messages.append(message)
        self.log_text.config(state='normal')  # Abilita temporaneamente
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')  # Disabilita di nuovo
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def load_api_keys_from_environment(self):
        """Carica le chiavi API dalle variabili d'ambiente."""
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_key.set(self.gemini_api_key)  # Carica nella GUI

    def start_processing(self):
        """Avvia l'elaborazione video in un thread separato."""
        if not self.ffmpeg_available:
            messagebox.showerror("Errore", "FFmpeg è richiesto!")
            return
        if not self.source_file.get() or not self.target_file.get():
            messagebox.showerror("Errore", "Seleziona i file di input e output.")
            return

        try:
            lang_str = self.lang_combo.get()
            self.target_language = next(
                (code for code, (name, flag) in self.languages.items() if f"{flag} {name}" == lang_str), None
            )
            if self.target_language is None:
                raise ValueError("Lingua non valida selezionata.")
        except ValueError as e:
            messagebox.showerror("Errore", str(e))
            return

        self.gemini_api_key = self.gemini_key.get()
        if self.gemini_api_key:
            self.init_gemini()
        else:
            self.log_message("Chiave API Gemini non fornita. Salto l'accorciamento.")

        self.start_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.log_text.config(state='normal')
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state='disabled')
        self.log_messages = []
        threading.Thread(target=self.process_video, daemon=True).start()

    def load_whisper_model(self):
        """Carica il modello Faster Whisper."""
        try:
            self.log_message(f"Caricamento del modello Whisper ({self.model_size}) su {self.device}...")
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.log_message("Modello Whisper caricato con successo.")
        except Exception as e:
            self.log_message(f"Errore durante il caricamento del modello Whisper: {e}")
            raise

    def extract_audio_in_chunks(self, video_path: str) -> Tuple[List[str], float]:
        """Estrae l'audio dal video in porzioni."""
        self.log_message(f"Estrazione audio da: {video_path}")
        audio_chunks = []
        try:
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Impossibile ottenere la durata del video")
            total_duration = float(duration_output.strip())
            self.log_message(f"Durata totale del video: {total_duration:.2f} secondi")

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
            self.log_message(f"Errore durante l'estrazione audio: {e}")
            raise  # Rilancia l'eccezione per essere gestita in process_video

    def transcribe_audio_in_chunks(self, audio_chunks: List[str], language: str) -> Tuple[List[Dict], int]:
        """Trascrive porzioni audio usando Faster Whisper."""
        all_segments = []
        total_words = 0
        if self.whisper_model is None:
            raise ValueError("Modello Whisper non caricato.")

        try:
            for i, chunk_path in enumerate(audio_chunks):
                self.log_message(f"Trascrizione porzione {i+1}/{len(audio_chunks)}: {chunk_path}")
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
                self.log_message(f"Trascrizione della porzione {i+1} completata.")
                self.root.update()  # Aggiorna GUI

                self.remove_file_with_retry(chunk_path)
                self.remove_directory_with_retry(os.path.dirname(chunk_path))

            self.log_message(f"Parole totali trascritte: {total_words}")
            return all_segments, total_words
        except Exception as e:
            self.log_message(f"Errore durante la trascrizione: {e}")
            raise

    def translate_and_refine_segments(self, segments: List[Dict]) -> Tuple[List[Dict], int]:
        """Traduce i segmenti e gestisce potenziali errori di traduzione."""
        translated_segments = []
        total_words_translated = 0
        translator = GoogleTranslator(source='auto', target=self.target_language)

        try:
            for segment in segments:
                self.log_message(f"Traduzione segmento: '{segment['text'][:50]}...' in {self.target_language}")
                translated_text = translator.translate(segment['text'])
                if not isinstance(translated_text, str):
                    self.log_message(
                        f"ATTENZIONE: translator.translate non ha restituito una stringa. Tipo: {type(translated_text)}, Valore: {translated_text}")
                    translated_text = ""  # Imposta a stringa vuota
                translated_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': translated_text
                })
                total_words_translated += len(translated_text.split())
                self.log_message(f"Segmento tradotto: '{translated_text[:50]}...'")
            return translated_segments, total_words_translated
        except Exception as e:
            self.log_message(f"Errore durante la traduzione: {e}")
            raise

    def generate_tts_audio_for_segments(self, translated_segments: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
        """Genera audio TTS per un batch di segmenti tradotti."""
        tts_chunks = []
        updated_segments = []
        total_duration = 0
        try:
            for i, segment in enumerate(translated_segments):
                text = segment['text']
                self.log_message(f"Generazione TTS per il segmento {i+1}/{len(translated_segments)}: '{text[:50]}...'")
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
                                                                        "Impossibile ottenere la durata del segmento TTS")
                    segment_duration = float(duration_output.strip())
                    updated_segments.append({
                        'start': total_duration,
                        'end': total_duration + segment_duration,
                        'text': text,
                    })
                    total_duration += segment_duration
                except Exception as e:
                    self.log_message(f"Errore durante la generazione TTS per il segmento {i + 1}: {e}")
                    # Pulisci tutti i file creati in questo batch finora
                    for file in tts_chunks:
                        self.remove_file_with_retry(file)
                        self.remove_directory_with_retry(os.path.dirname(file))  # Prova a rimuovere la directory
                    return None, []  # Indica fallimento

                self.log_message(f"TTS per il segmento {i+1} generato.")
                self.root.update()  # Mantieni la GUI reattiva

            self.log_message(f"Unione di {len(tts_chunks)} porzioni TTS...")
            if tts_chunks:
                merged_audio_path = os.path.join(self.process_temp_dir, f"merged_tts_audio_{int(time.time())}.mp3")
                self.merge_audio_chunks(tts_chunks, merged_audio_path)
                self.log_message(f"Audio TTS unito: {merged_audio_path}")

                for tts_file in tts_chunks:
                    self.remove_file_with_retry(tts_file)
                    self.remove_directory_with_retry(os.path.dirname(tts_file))

                return merged_audio_path, updated_segments
            else:
                self.log_message("Nessun segmento TTS generato.")
                return None, []

        except Exception as e:
            self.log_message(f"Errore durante la generazione/unione TTS: {e}")
            for file in tts_chunks:  # Corretto
                self.remove_file_with_retry(file)
                self.remove_directory_with_retry(os.path.dirname(file))  # e la directory
            return None, []

    def open_webpage(self, url):
        """Apre una pagina web nel browser predefinito."""
        import webbrowser
        webbrowser.open(url)

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Download del tokenizer punkt NLTK in corso...")
        nltk.download('punkt')
    app = LinguoAIVideoVoicePro()
    app.root.mainloop()