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
        # --- Paramètres de configuration ---
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
        self.hw_accel_info = self.detect_hardware_acceleration()  # Enregistrer les informations
        self.hw_accel = self.hw_accel_info['accel'] if self.hw_accel_info else None

        # --- Statut interne ---
        self.whisper_model: Optional[WhisperModel] = None
        self.current_process: Optional[subprocess.Popen] = None
        self.process_start_time: Optional[float] = None
        self.process_temp_dir: str = ""

        # --- Composants GUI ---
        self.root = ttk.Window(themename="darkly")
        self.root.title("LinguoAI VideoVoice Pro")
        self.root.geometry("640x850")  # Définir la taille initiale

        self.source_file = tk.StringVar()
        self.target_file = tk.StringVar()
        self.selected_language = tk.StringVar(value="en")
        self.gemini_key = tk.StringVar()  # Clé API Gemini
        self.progress_var = tk.DoubleVar(value=0)
        self.log_messages: List[str] = []

        self.languages = {
            "af": ("Afrikaans", "🇿🇦"),
            "sq": ("Albanais", "🇦🇱"),
            "am": ("Amharique", "🇪🇹"),
            "ar": ("Arabe", "🇸🇦"),
            "eu": ("Basque", "🇪🇸"),
            "bn": ("Bengali", "🇧🇩"),
            "bs": ("Bosniaque", "🇧🇦"),
            "bg": ("Bulgare", "🇧🇬"),
            "ca": ("Catalan", "🇦🇩"),
            "zh-CN": ("Chinois (Simplifié)", "🇨🇳"),
            "zh-TW": ("Chinois (Traditionnel)", "🇹🇼"),
            "hr": ("Croate", "🇭🇷"),
            "cs": ("Tchèque", "🇨🇿"),
            "da": ("Danois", "🇩🇰"),
            "nl": ("Néerlandais", "🇳🇱"),
            "en": ("Anglais", "🇬🇧"),
            "et": ("Estonien", "🇪🇪"),
            "tl": ("Philippin", "🇵🇭"),
            "fi": ("Finlandais", "🇫🇮"),
            "fr": ("Français", "🇫🇷"),
            "gl": ("Galicien", "🇪🇸"),
            "de": ("Allemand", "🇩🇪"),
            "el": ("Grec", "🇬🇷"),
            "gu": ("Gujarati", "🇮🇳"),
            "ha": ("Haoussa", "🇳🇬"),
            "he": ("Hébreu", "🇮🇱"),
            "hi": ("Hindi", "🇮🇳"),
            "hu": ("Hongrois", "🇭🇺"),
            "is": ("Islandais", "🇮🇸"),
            "id": ("Indonésien", "🇮🇩"),
            "it": ("Italien", "🇮🇹"),
            "ja": ("Japonais", "🇯🇵"),
            "jw": ("Javanais", "🇮🇩"),
            "kn": ("Kannada", "🇮🇳"),
            "km": ("Khmer", "🇰🇭"),
            "ko": ("Coréen", "🇰🇷"),
            "la": ("Latin", "🇻🇦"),
            "lv": ("Letton", "🇱🇻"),
            "lt": ("Lituanien", "🇱🇹"),
            "ms": ("Malais", "🇲🇾"),
            "mr": ("Marathi", "🇮🇳"),
            "ml": ("Malayalam", "🇮🇳"),
            "my": ("Myanmar (Birman)", "🇲🇲"),
            "ne": ("Népalais", "🇳🇵"),
            "no": ("Norvégien", "🇳🇴"),
            "pa": ("Pendjabi", "🇮🇳"),
            "pl": ("Polonais", "🇵🇱"),
            "pt": ("Portugais", "🇵🇹"),
            "ro": ("Roumain", "🇷🇴"),
            "ru": ("Russe", "🇷🇺"),
            "sr": ("Serbe", "🇷🇸"),
            "si": ("Cingalais", "🇱🇰"),
            "sk": ("Slovaque", "🇸🇰"),
            "sl": ("Slovène", "🇸🇮"),
            "es": ("Espagnol", "🇪🇸"),
            "su": ("Soundanais", "🇮🇩"),
            "sw": ("Swahili", "🇰🇪"),
            "sv": ("Suédois", "🇸🇪"),
            "ta": ("Tamil", "🇮🇳"),
            "te": ("Télougou", "🇮🇳"),
            "th": ("Thaï", "🇹🇭"),
            "tr": ("Turc", "🇹🇷"),
            "uk": ("Ukrainien", "🇺🇦"),
            "ur": ("Ourdou", "🇵🇰"),
            "vi": ("Vietnamien", "🇻🇳"),
            "cy": ("Gallois", "🇬🇧")
        }

        # --- Initialisation ---
        self.setup_gui()
        self.setup_ffmpeg()
        self.load_api_keys_from_environment()
        self.load_whisper_model()
        if self.gemini_api_key:
            self.init_gemini()
        self.log_hardware_acceleration()

    def seconds_to_srt_time(self, seconds: float) -> str:
        """Convertit les secondes au format temporel SRT (HH:MM:SS,mmm)."""
        milliseconds = int((seconds * 1000) % 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    def create_srt_file(self, segments: List[Dict], output_path: str):
        """Crée un fichier SRT à partir des segments transcrits/traduits."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments):
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text']

                    # Convertir les secondes au format temporel SRT
                    start_time_srt = self.seconds_to_srt_time(start_time)
                    end_time_srt = self.seconds_to_srt_time(end_time)

                    f.write(f"{i + 1}\n")
                    f.write(f"{start_time_srt} --> {end_time_srt}\n")
                    f.write(f"{text}\n\n")

            self.log_message(f"Fichier SRT créé : {output_path}")

        except Exception as e:
            self.log_message(f"Erreur lors de la création du fichier SRT : {e}")

    def log_hardware_acceleration(self):
        """Enregistre les informations d'accélération matérielle."""
        if self.hw_accel_info:
            self.log_message(f"Accélération matérielle détectée : {self.hw_accel_info['accel']} ({self.hw_accel_info['info']})")
        else:
            self.log_message("Aucune accélération matérielle détectée.")

    def detect_hardware_acceleration(self):
        """Détecte l'accélération matérielle (NVIDIA, Intel, AMD)."""
        try:
            # NVIDIA
            try:
                subprocess.run(['nvidia-smi'], check=True, capture_output=True)
                return {'accel': 'cuda', 'info': 'GPU NVIDIA détectée'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # Intel (Quick Sync)
            try:
                output = subprocess.run(['vainfo'], check=True, capture_output=True, text=True).stdout
                if "VA-API" in output:  # Vérification très grossière, peut être affinée
                    return {'accel': 'qsv', 'info': 'Intel Quick Sync détecté'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # AMD (AMF)  - un peu plus complexe, car il n'y a pas de simple "amfinfo"
            #  On pourrait rechercher des pilotes/périphériques spécifiques, mais c'est spécifique au système d'exploitation.
            #  Voici une vérification très simple et incomplète pour Linux :
            if os.name == 'posix':  # Linux/macOS
                try:
                    output = subprocess.run(['lspci', '-v'], check=True, capture_output=True, text=True).stdout
                    if "Advanced Micro Devices, Inc. [AMD/ATI]" in output:
                        return {'accel': 'h264_vaapi', 'info': 'GPU AMD détectée (VAAPI)'}  # Supposition !
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass

            return None  # Aucune accélération matérielle trouvée

        except Exception as e:
            self.log_message(f"Erreur lors de la détection de l'accélération matérielle : {e}")
            return None
    def init_gemini(self):
        """Initialise le modèle Gemini Pro."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.log_message("Modèle Gemini Pro initialisé.")
        except Exception as e:
            self.log_message(f"Erreur lors de l'initialisation de Gemini Pro : {e}")
            messagebox.showerror("Erreur Gemini", f"Impossible d'initialiser Gemini Pro : {e}")
            self.model = None  # Définir sur None en cas d'échec

    def check_process_timeout(self):
        """Vérifie si le processus global a dépassé le temps maximal autorisé."""
        if self.process_start_time and time.time() - self.process_start_time > self.process_timeout:
            if self.current_process:
                self.log_message(f"Délai d'attente du processus dépassé ! Arrêt du processus (PID : {self.current_process.pid})...")
                self.kill_process_tree(self.current_process.pid)  # Arrêter l'arborescence des processus !
            raise TimeoutError("Le processus a dépassé le temps maximal autorisé")

    def kill_process_tree(self, pid):
        """Arrête un processus et tous ses processus enfants."""
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):  # Obtenir tous les enfants/petits-enfants
                self.log_message(f"Arrêt du processus enfant : {child.pid}")
                child.kill()
            self.log_message(f"Arrêt du processus parent : {parent.pid}")
            parent.kill()
        except psutil.NoSuchProcess:
            self.log_message(f"Processus avec PID {pid} introuvable.")
        except Exception as e:
            self.log_message(f"Erreur lors de l'arrêt de l'arborescence des processus : {e}")

    def run_subprocess_with_timeout(self, command, timeout, error_message):
        """Exécute un sous-processus avec un délai d'attente dynamique et des vérifications de vitalité."""
        try:
            self.log_message(f"Exécution de la commande avec un délai d'attente de {timeout} : {' '.join(command)}")
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.log_message(f"Processus démarré avec le PID : {self.current_process.pid}")

            start_time = time.time()
            last_output_time = start_time

            stdout, stderr = self.current_process.communicate(timeout=timeout)  # Utiliser communicate !
            retcode = self.current_process.returncode

            if retcode != 0:
                self.log_message(f"Le processus a échoué avec le code d'erreur {retcode} :")
                self.log_message(f"Stdout : {stdout}")
                self.log_message(f"Stderr : {stderr}")
                raise subprocess.CalledProcessError(retcode, command, stdout, stderr)

            self.log_message("Commande exécutée avec succès.")
            return stdout, stderr

        except subprocess.TimeoutExpired:
            self.log_message(f"Le processus a expiré après {timeout} secondes")
            self.kill_process_tree(self.current_process.pid)  # Arrêter l'arborescence des processus !
            stdout, stderr = self.current_process.communicate()  # Obtenir toute sortie restante
            self.log_message(f"Stdout : {stdout}")
            self.log_message(f"Stderr : {stderr}")
            raise TimeoutError(error_message)

        except Exception as e:
            self.log_message(f"Une erreur inattendue s'est produite : {e}")
            if self.current_process:
                self.kill_process_tree(self.current_process.pid)  # Arrêter si toujours en cours d'exécution
            raise
        finally:
            self.current_process = None

    def extract_audio_chunk(self, video_path, audio_path, start_time, duration):
        """Extrait un morceau d'audio de la vidéo."""
        command = [
            "ffmpeg",
            "-y",  # Écraser les fichiers de sortie sans demander
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-vn",  # Désactiver la vidéo
            "-acodec", "libmp3lame",
            "-q:a", "2",  # MP3 de bonne qualité
            "-loglevel", "error",  # Réduire la verbosité
            audio_path
        ]
        if self.hw_accel:
            command.insert(1, "-hwaccel")
            command.insert(2, self.hw_accel)

        try:
            self.run_subprocess_with_timeout(command, self.chunk_timeout, "L'extraction audio a expiré")
            self.log_message(f"Morceau audio extrait : '{audio_path}'")
        except Exception as e:
            self.log_message(f"Erreur lors de l'extraction du morceau audio : {e}")
            raise

    def batch_segments(self, segments: List[Dict]) -> List[List[Dict]]:
        """Divise les segments en plus petits lots pour la TTS."""
        batched_segments = []
        for i in range(0, len(segments), self.tts_batch_size):
            batch = segments[i:i + self.tts_batch_size]
            batched_segments.append(batch)
        return batched_segments

    def validate_audio_chunk(self, chunk_path: str) -> bool:
        """Valide un morceau audio en utilisant ffprobe."""
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
                f"La validation du morceau audio a expiré : {chunk_path}"
            )
            if stdout.strip():
                self.log_message(f"Morceau audio validé : {chunk_path}")
                return True
            else:
                self.log_message(f"La validation du morceau audio a échoué (pas de durée) : {chunk_path}")
                return False
        except Exception as e:
            self.log_message(f"Erreur lors de la validation du morceau audio {chunk_path} : {e}")
            return False

    def merge_audio_chunks(self, audio_chunks, output_path):
        """Fusionne plusieurs morceaux audio en un seul fichier en utilisant FFmpeg."""
        if not audio_chunks:
            self.log_message("Aucun morceau audio à fusionner.")
            return

        valid_chunks = [chunk for chunk in audio_chunks if self.validate_audio_chunk(chunk)]
        if not valid_chunks:
            self.log_message("Aucun morceau audio valide à fusionner.")
            return

        timestamp = int(time.time())
        temp_dir = os.path.join(tempfile.gettempdir(), f"audio_merge_temp_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)

        list_file_path = os.path.join(temp_dir, "chunk_list.txt")
        try:
            with open(list_file_path, "w") as f:
                for chunk_path in valid_chunks:
                    abs_chunk_path = os.path.abspath(chunk_path)  # Utiliser le chemin absolu
                    f.write(f"file '{abs_chunk_path}'\n")

            command = [
                "ffmpeg",
                "-y",  # Écraser
                "-f", "concat",
                "-safe", "0",  # Requis pour les chemins absolus avec concat
                "-i", list_file_path,
                "-c", "copy",
                "-loglevel", "error",
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            # Calculer un délai d'attente dynamique basé sur le nombre de morceaux.
            merge_timeout = len(valid_chunks) * 10 + 30  # 10 secondes par morceau + 30 de base
            self.run_subprocess_with_timeout(command, merge_timeout, "La fusion audio a expiré")
            self.log_message(f"Morceaux audio fusionnés : '{output_path}'")

        except Exception as e:
            self.log_message(f"Erreur lors de la fusion des morceaux audio : {e}")
            raise
        finally:
            self.remove_file_with_retry(list_file_path)  # Utiliser la fonction de nouvelle tentative
            self.remove_directory_with_retry(temp_dir)  # et pour le répertoire

    def merge_video_audio(self, audio_file):
        """Fusionne l'audio final avec la vidéo originale."""
        try:
            output_path = self.target_file.get()
            video_path = self.source_file.get()

            # Obtenir la durée de la vidéo en utilisant ffprobe
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Impossible d'obtenir la durée de la vidéo")
            total_duration = float(duration_output.strip())
            self.log_message(f"Durée de la vidéo pour la fusion : {total_duration:.2f} secondes")

            # Calculer le délai d'attente dynamique (par exemple, 3x la durée + 120 secondes)
            dynamic_timeout = int(3 * total_duration + 120)
            self.log_message(f"Délai d'attente dynamique pour la fusion : {dynamic_timeout} secondes")

            command = [
                'ffmpeg',
                '-y',  # Écraser le fichier de sortie
                '-i', video_path,
                '-i', audio_file,
                '-c:v', 'h264_nvenc' if self.hw_accel == 'cuda' else 'libx264',  # H.265
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',  # Terminer l'encodage lorsque le flux le plus court se termine
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            self.run_subprocess_with_timeout(command, dynamic_timeout, "La fusion vidéo et audio a expiré")
            self.log_message(f"Vidéo et audio fusionnés : {output_path}")
        except Exception as e:
            self.log_message(f"Erreur lors de la fusion de la vidéo et de l'audio : {e}")
            raise

    def shorten_text_with_gemini(self, text: str) -> str:
        """Raccourcit le texte en utilisant Gemini Pro, en gérant les erreurs et les limites de débit."""
        if self.model is None:
            self.log_message("Modèle Gemini Pro non initialisé. Raccourcissement ignoré.")
            return text

        try:
            prompt = f"Veuillez raccourcir le texte suivant tout en préservant les informations clés :\n\n{text}"
            time.sleep(1.5)  # Limitation du débit : Pause de 1,5 seconde
            response = self.model.generate_content(prompt)
            if response and response.text:
                shortened_text = response.text
                self.log_message("Texte raccourci avec Gemini Pro.")
                return shortened_text
            else:
                self.log_message("Gemini Pro a renvoyé une réponse vide. Utilisation du texte original.")
                return text
        except Exception as e:
            self.log_message(f"Erreur lors du raccourcissement du texte avec Gemini : {e}")
            return text

    def process_video(self):
        """Flux de travail principal du traitement vidéo."""
        self.process_start_time = time.time()
        timestamp = int(time.time())
        self.process_temp_dir = os.path.join(tempfile.gettempdir(), f"process_temp_{timestamp}")
        os.makedirs(self.process_temp_dir, exist_ok=True)

        try:
            self.log_message("Démarrage du traitement vidéo...")
            self.progress_var.set(5)

            # Configurer la vérification du délai d'attente *avant* de commencer toute tâche.
            def check_timeout():
                self.check_process_timeout()
                self.root.after(1000, check_timeout)  # Vérifier toutes les secondes

            check_timeout()  # Démarrer le vérificateur de délai d'attente

            self.progress_var.set(10)
            audio_chunks, total_duration = self.extract_audio_in_chunks(self.source_file.get())
            self.progress_var.set(25)

            segments, total_words_original = self.transcribe_audio_in_chunks(audio_chunks, self.target_language)
            if not segments:
                raise Exception("La transcription a échoué.")
            self.progress_var.set(45)

            translated_segments, total_words_translated = self.translate_and_refine_segments(segments)

            # --- Raccourcissement de texte Gemini (Optionnel) ---
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
                    self.root.update()  # Mettre à jour l'interface graphique
                translated_segments = shortened_segments

                shortening_change = (
                    ((total_words_translated - total_words_shortened) / total_words_translated) * 100
                    if total_words_translated else 0
                )
                self.log_message(f"Le raccourcissement a réduit le nombre de mots de : {shortening_change:.2f}%")
            else:
                total_words_shortened = total_words_translated
                self.log_message("Clé API Gemini non fournie. Raccourcissement ignoré.")

            self.progress_var.set(60)

            # Créer le fichier SRT (exemple)
            srt_output_path = os.path.splitext(self.target_file.get())[0] + ".srt"  # Correspond au nom du fichier vidéo de sortie
            self.create_srt_file(translated_segments, srt_output_path)

            # --- Rapport de nombre de mots ---
            if total_words_original > 0:
                translation_change = (total_words_translated / total_words_original) * 100
                self.log_message(f"Nombre de mots original : {total_words_original}")
                self.log_message(f"Nombre de mots traduits : {total_words_translated}")
                self.log_message(f"Changement du nombre de mots traduit : {translation_change:.2f}%")
            else:
                self.log_message("Le nombre de mots original est zéro. Pourcentage ignoré.")

            # --- TTS et fusion audio ---
            batched_translated_segments = self.batch_segments(translated_segments)
            all_updated_segments = []
            merged_audio_files = []

            for i, batch in enumerate(batched_translated_segments):
                self.log_message(f"Traitement du lot TTS {i+1}/{len(batched_translated_segments)}")
                merged_audio_path, updated_segments = self.generate_tts_audio_for_segments(batch)
                if merged_audio_path:
                    all_updated_segments.extend(updated_segments)
                    merged_audio_files.append(merged_audio_path)
                else:
                    self.log_message(f"Le lot TTS {i+1} a échoué.")
                    # Nettoyer *tous* les fichiers TTS créés précédemment en cas d'échec
                    for file_path in merged_audio_files:
                        self.remove_file_with_retry(file_path)
                    raise Exception(f"La génération audio TTS a échoué pour le lot {i + 1}.")

            # --- Fusion audio finale (si plusieurs lots) ---
            if len(merged_audio_files) > 1:
                final_merged_audio_path = os.path.join(self.process_temp_dir,
                                                        f"final_merged_tts_audio_{timestamp}.mp3")
                self.merge_audio_chunks(merged_audio_files, final_merged_audio_path)
                self.log_message(f"Audio TTS final fusionné : {final_merged_audio_path}")
            elif merged_audio_files:
                final_merged_audio_path = merged_audio_files[0]
                self.log_message(f"Audio TTS final fusionné (lot unique) : {final_merged_audio_path}")
            else:
                raise Exception("Aucun audio TTS généré.")

            # --- Nettoyer les fichiers TTS intermédiaires ---
            for file_path in merged_audio_files:
                if file_path != final_merged_audio_path:  # Ne pas supprimer le fichier final !
                    self.remove_file_with_retry(file_path)

            self.progress_var.set(80)

            # --- Fusion vidéo/audio finale ---
            self.merge_video_audio(final_merged_audio_path)
            self.progress_var.set(95)

            self.log_message("Traitement terminé avec succès ! 🎉")
            messagebox.showinfo("Succès", "Traitement vidéo terminé !")


        except TimeoutError as e:
            self.log_message(f"Le traitement a expiré : {str(e)}")
            messagebox.showerror("Erreur", f"Le traitement a expiré : {str(e)}")
        except Exception as e:
            self.log_message(f"Erreur : {str(e)}")
            messagebox.showerror("Erreur", f"Le traitement a échoué : {str(e)}")
        finally:
            # --- Nettoyage ---
            self.progress_var.set(0)
            self.process_start_time = None
            self.current_process = None
            self.start_button.config(state=tk.NORMAL)  # Réactiver le bouton
            self.remove_directory_with_retry(self.process_temp_dir)

    def remove_file_with_retry(self, file_path, retries=3, delay=0.5):
        """Supprime un fichier, en réessayant si nécessaire."""
        file_path = os.path.abspath(file_path)  # Utiliser le chemin absolu
        for i in range(retries):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                self.log_message(f"Fichier supprimé : {file_path}")
                return  # Succès
            except OSError as e:
                if e.errno == errno.ENOENT:  # Fichier introuvable - déjà disparu
                    self.log_message(f"Fichier introuvable (déjà supprimé) : {file_path}")
                    return
                if i < retries - 1:  # Ne pas attendre lors de la dernière tentative
                    self.log_message(f"Nouvelle tentative de suppression de fichier ({i+1}/{retries}) : {file_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Erreur lors de la suppression du fichier après plusieurs tentatives : {file_path} - {e}")
            except Exception as e:
                self.log_message(f"Erreur inattendue lors de la suppression du fichier : {file_path} - {e}")
                return  # Ne pas réessayer pour les erreurs inattendues

    def remove_directory_with_retry(self, dir_path, retries=5, delay=1):
        """Supprime un répertoire, en réessayant si nécessaire (surtout pour les non-vides)."""
        dir_path = os.path.abspath(dir_path)
        for i in range(retries):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                self.log_message(f"Répertoire supprimé : {dir_path}")
                return
            except OSError as e:
                if e.errno == errno.ENOENT:  # Répertoire déjà supprimé
                    self.log_message(f"Répertoire déjà supprimé : {dir_path}")
                    return
                elif e.errno == errno.ENOTEMPTY:  # Répertoire non vide
                    self.log_message(f"Répertoire non vide, nouvelle tentative de suppression ({i+1}/{retries}) : {dir_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Erreur lors de la suppression du répertoire : {dir_path} - {e}")
                    time.sleep(delay)  # Attendre même pour les autres erreurs
            except Exception as e:
                self.log_message(f"Erreur inattendue lors de la suppression du répertoire : {dir_path} - {e}")
                return

    def setup_ffmpeg(self):
        """Vérifie si FFmpeg est disponible."""
        try:
            self.run_subprocess_with_timeout(['ffmpeg', '-version'], 10, "La vérification de FFmpeg a échoué")
            self.ffmpeg_available = True
            self.log_message("FFmpeg détecté.")
        except FileNotFoundError:
            self.ffmpeg_available = False
            self.log_message("FFmpeg introuvable. Veuillez installer FFmpeg.")
            messagebox.showwarning("FFmpeg manquant", "FFmpeg est requis. Veuillez l'installer.")
        except Exception as e:
            self.ffmpeg_available = False
            self.log_message(f"La vérification de FFmpeg a échoué : {e}")
            messagebox.showwarning("Erreur FFmpeg", "La vérification de FFmpeg a échoué. Vérifiez l'installation.")

    def setup_gui(self):
        """Configure l'interface utilisateur graphique."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- En-tête ---
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        title_label = ttk.Label(header_frame, text="🎬 LinguoAI VideoVoice Pro", font=('Helvetica', 16, 'bold'))
        title_label.pack()

        # --- Sélection de fichiers ---
        file_frame = ttk.LabelFrame(main_frame, text="Fichiers vidéo", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        # Utiliser une disposition en grille pour les lignes d'entrée/sortie
        source_frame = ttk.Frame(file_frame)
        source_frame.pack(fill=tk.X, pady=2)
        ttk.Label(source_frame, text="📹 Entrée :", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_file, width=40)
        self.source_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(source_frame, text="Parcourir", command=self.browse_source, style="Accent.TButton").grid(row=0, column=2, padx=5)

        target_frame = ttk.Frame(file_frame)
        target_frame.pack(fill=tk.X, pady=2)
        ttk.Label(target_frame, text="💾 Sortie :", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.target_entry = ttk.Entry(target_frame, textvariable=self.target_file, width=40)
        self.target_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(target_frame, text="Parcourir", command=self.browse_target, style="Accent.TButton").grid(row=0, column=2, padx=5)

        # Rendre les colonnes d'entrée extensibles
        source_frame.columnconfigure(1, weight=1)
        target_frame.columnconfigure(1, weight=1)


        # --- Sélection de la langue ---
        lang_frame = ttk.LabelFrame(main_frame, text="Paramètres vocaux", padding="10")
        lang_frame.pack(fill=tk.X, pady=5)

        lang_combo_frame = ttk.Frame(lang_frame)
        lang_combo_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lang_combo_frame, text="🗣️ Langue cible :").pack(side=tk.LEFT, padx=5)

        # Combobox avec recherche
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
        self.lang_combo.set("🇬🇧 Anglais")  # Définir la valeur par défaut après la création de la combobox
        self.language_search_entry.bind("<Return>", (lambda event: self.lang_combo.focus()))
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_language_selected)

        # --- Options de fusion ---
        options_frame = ttk.LabelFrame(main_frame, text="Options de fusion", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        # Clé API Gemini
        brain_frame = ttk.Frame(options_frame)
        brain_frame.pack(fill=tk.X, pady=2)

        # --- Description de la clé API Gemini ---
        gemini_description_label = ttk.Label(
            brain_frame,
            wraplength=600,  # Retour à la ligne du texte
            justify=tk.LEFT,  # Justifier le texte à gauche
            text="Ce programme utilise l'API Google Gemini Pro pour le raccourcissement de texte optionnel.  "
                 "Cela peut aider à réduire la longueur globale du texte traduit tout en préservant les informations clés.\n"
                 "Une clé API est requise pour utiliser cette fonctionnalité. Si vous n'avez pas de clé, vous pouvez ignorer cette étape, "
                 "et le programme continuera sans raccourcissement."
        )
        gemini_description_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        gemini_link_label = ttk.Label(
            brain_frame,
            text="Obtenez une clé API Gemini ici : ",
            foreground="dodgerblue",  # Changé en un bleu plus subtil
            cursor="hand2"  # Changer le curseur au survol
        )
        gemini_link_label.pack(side=tk.TOP, fill=tk.X)
        gemini_link_label.bind("<Button-1>", lambda e: self.open_webpage("https://makersuite.google.com/app/apikey"))

        self.gemini_key_label = ttk.Label(brain_frame, text="Clé API Gemini :")
        self.gemini_key_label.pack(side=tk.LEFT, padx=5)
        self.gemini_key_entry = ttk.Entry(brain_frame, textvariable=self.gemini_key, width=30, show="*")
        self.gemini_key_entry.pack(side=tk.LEFT)

        self.start_button = ttk.Button(options_frame, text="🚀 Démarrer le traitement", command=self.start_processing,
                                       style="Success.TButton")
        self.start_button.pack(pady=10)

        # --- Progression ---
        progress_frame = ttk.LabelFrame(main_frame, text="Progression", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100,
                                            style="Success.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(progress_frame, text="Prêt")
        self.status_label.pack()

        # --- Journal ---
        log_frame = ttk.LabelFrame(main_frame, text="Journal de traitement", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, state='disabled', wrap=tk.WORD)  # Activer le retour à la ligne
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def on_language_selected(self, event):
        """Gère la sélection de la langue depuis la combobox."""
        selected_lang_str = self.lang_combo.get()
        for code, (name, flag) in self.languages.items():
            if f"{flag} {name}" == selected_lang_str:
                self.target_language = code  # Mettre à jour la langue cible
                self.log_message(f"Langue cible sélectionnée : {name} ({code})")
                break

    def update_language_list(self, *args):
        """Filtre la liste des langues en fonction de la saisie de recherche."""
        search_term = self.language_search_var.get().lower()
        filtered_languages = [
            f"{flag} {name}"
            for code, (name, flag) in self.languages.items()
            if search_term in name.lower() or search_term in code.lower() or search_term in flag.lower()
        ]
        self.lang_combo['values'] = filtered_languages
        if filtered_languages:
            self.lang_combo.set(filtered_languages[0])  # Définir sur la première correspondance
        else:
            self.lang_combo.set('')  # Effacer si aucune correspondance
    def browse_source(self):
        """Ouvre une boîte de dialogue de fichier pour sélectionner la vidéo source."""
        filename = filedialog.askopenfilename(
            title="Sélectionner un fichier vidéo",
            filetypes=[("Fichiers vidéo", "*.mp4 *.avi *.mkv *.mov"), ("Tous les fichiers", "*.*")]
        )
        if filename:
            self.source_file.set(filename)
            base = os.path.splitext(filename)[0]
            self.target_file.set(f"{base}_translated.mp4")
            self.source_entry.xview_moveto(1) # Faire défiler jusqu'à la fin
            self.target_entry.xview_moveto(1)

    def browse_target(self):
        """Ouvre une boîte de dialogue de fichier pour sélectionner le chemin de la vidéo cible."""
        filename = filedialog.asksaveasfilename(
            title="Enregistrer la vidéo traduite",
            defaultextension=".mp4",
            filetypes=[("Fichiers MP4", "*.mp4")]
        )
        if filename:
            self.target_file.set(filename)
            self.source_entry.xview_moveto(1)  # Faire défiler jusqu'à la fin
            self.target_entry.xview_moveto(1)
    def log_message(self, message):
        """Enregistre un message dans l'interface graphique et la liste de journaux interne."""
        self.log_messages.append(message)
        self.log_text.config(state='normal')  # Activer temporairement
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')  # Désactiver à nouveau
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def load_api_keys_from_environment(self):
        """Charge les clés API depuis les variables d'environnement."""
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_key.set(self.gemini_api_key)  # Charger dans l'interface graphique

    def start_processing(self):
        """Démarre le traitement vidéo dans un thread séparé."""
        if not self.ffmpeg_available:
            messagebox.showerror("Erreur", "FFmpeg est requis !")
            return
        if not self.source_file.get() or not self.target_file.get():
            messagebox.showerror("Erreur", "Sélectionnez les fichiers d'entrée et de sortie.")
            return

        try:
            lang_str = self.lang_combo.get()
            self.target_language = next(
                (code for code, (name, flag) in self.languages.items() if f"{flag} {name}" == lang_str), None
            )
            if self.target_language is None:
                raise ValueError("Langue invalide sélectionnée.")
        except ValueError as e:
            messagebox.showerror("Erreur", str(e))
            return

        self.gemini_api_key = self.gemini_key.get()
        if self.gemini_api_key:
            self.init_gemini()
        else:
            self.log_message("Clé API Gemini non fournie. Raccourcissement ignoré.")

        self.start_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.log_text.config(state='normal')
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state='disabled')
        self.log_messages = []
        threading.Thread(target=self.process_video, daemon=True).start()

    def load_whisper_model(self):
        """Charge le modèle Faster Whisper."""
        try:
            self.log_message(f"Chargement du modèle Whisper ({self.model_size}) sur {self.device}...")
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.log_message("Modèle Whisper chargé avec succès.")
        except Exception as e:
            self.log_message(f"Erreur lors du chargement du modèle Whisper : {e}")
            raise

    def extract_audio_in_chunks(self, video_path: str) -> Tuple[List[str], float]:
        """Extrait l'audio de la vidéo par morceaux."""
        self.log_message(f"Extraction audio de : {video_path}")
        audio_chunks = []
        try:
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Impossible d'obtenir la durée de la vidéo")
            total_duration = float(duration_output.strip())
            self.log_message(f"Durée totale de la vidéo : {total_duration:.2f} secondes")

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
            self.log_message(f"Erreur lors de l'extraction audio : {e}")
            raise  # Relancer l'exception pour qu'elle soit gérée dans process_video

    def transcribe_audio_in_chunks(self, audio_chunks: List[str], language: str) -> Tuple[List[Dict], int]:
        """Transcrire les morceaux audio en utilisant Faster Whisper."""
        all_segments = []
        total_words = 0
        if self.whisper_model is None:
            raise ValueError("Modèle Whisper non chargé.")

        try:
            for i, chunk_path in enumerate(audio_chunks):
                self.log_message(f"Transcription du morceau {i+1}/{len(audio_chunks)} : {chunk_path}")
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
                self.log_message(f"Transcription du morceau {i+1} terminée.")
                self.root.update()  # Mettre à jour l'interface graphique

                self.remove_file_with_retry(chunk_path)
                self.remove_directory_with_retry(os.path.dirname(chunk_path))

            self.log_message(f"Nombre total de mots transcrits : {total_words}")
            return all_segments, total_words
        except Exception as e:
            self.log_message(f"Erreur lors de la transcription : {e}")
            raise

    def translate_and_refine_segments(self, segments: List[Dict]) -> Tuple[List[Dict], int]:
        """Traduit les segments et gère les erreurs de traduction potentielles."""
        translated_segments = []
        total_words_translated = 0
        translator = GoogleTranslator(source='auto', target=self.target_language)

        try:
            for segment in segments:
                self.log_message(f"Traduction du segment : '{segment['text'][:50]}...' vers {self.target_language}")
                translated_text = translator.translate(segment['text'])
                if not isinstance(translated_text, str):
                    self.log_message(
                        f"AVERTISSEMENT : translator.translate n'a pas renvoyé une chaîne de caractères. Type : {type(translated_text)}, Valeur : {translated_text}")
                    translated_text = ""  # Définir sur une chaîne vide
                translated_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': translated_text
                })
                total_words_translated += len(translated_text.split())
                self.log_message(f"Segment traduit : '{translated_text[:50]}...'")
            return translated_segments, total_words_translated
        except Exception as e:
            self.log_message(f"Erreur lors de la traduction : {e}")
            raise

    def generate_tts_audio_for_segments(self, translated_segments: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
        """Génère l'audio TTS pour un lot de segments traduits."""
        tts_chunks = []
        updated_segments = []
        total_duration = 0
        try:
            for i, segment in enumerate(translated_segments):
                text = segment['text']
                self.log_message(f"Génération TTS pour le segment {i+1}/{len(translated_segments)} : '{text[:50]}...'")
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
                                                                        "Impossible d'obtenir la durée du segment TTS")
                    segment_duration = float(duration_output.strip())
                    updated_segments.append({
                        'start': total_duration,
                        'end': total_duration + segment_duration,
                        'text': text,
                    })
                    total_duration += segment_duration
                except Exception as e:
                    self.log_message(f"Erreur lors de la génération TTS pour le segment {i + 1} : {e}")
                    # Nettoyer tous les fichiers créés dans ce lot jusqu'à présent
                    for file in tts_chunks:
                        self.remove_file_with_retry(file)
                        self.remove_directory_with_retry(os.path.dirname(file))  # Essayer de supprimer le répertoire
                    return None, []  # Indiquer l'échec

                self.log_message(f"TTS pour le segment {i+1} généré.")
                self.root.update()  # Maintenir l'interface graphique réactive

            self.log_message(f"Fusion de {len(tts_chunks)} morceaux TTS...")
            if tts_chunks:
                merged_audio_path = os.path.join(self.process_temp_dir, f"merged_tts_audio_{int(time.time())}.mp3")
                self.merge_audio_chunks(tts_chunks, merged_audio_path)
                self.log_message(f"Audio TTS fusionné : {merged_audio_path}")

                for tts_file in tts_chunks:
                    self.remove_file_with_retry(tts_file)
                    self.remove_directory_with_retry(os.path.dirname(tts_file))

                return merged_audio_path, updated_segments
            else:
                self.log_message("Aucun segment TTS généré.")
                return None, []

        except Exception as e:
            self.log_message(f"Erreur lors de la génération/fusion TTS : {e}")
            for file in tts_chunks:  # Corrigé
                self.remove_file_with_retry(file)
                self.remove_directory_with_retry(os.path.dirname(file))  # et le répertoire
            return None, []

    def open_webpage(self, url):
        """Ouvre une page web dans le navigateur par défaut."""
        import webbrowser
        webbrowser.open(url)

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Téléchargement du tokenizer punkt NLTK...")
        nltk.download('punkt')
    app = LinguoAIVideoVoicePro()
    app.root.mainloop()