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
        # --- Параметри конфігурації ---
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
        self.hw_accel_info = self.detect_hardware_acceleration()  # Збереження інформації
        self.hw_accel = self.hw_accel_info['accel'] if self.hw_accel_info else None

        # --- Внутрішній статус ---
        self.whisper_model: Optional[WhisperModel] = None
        self.current_process: Optional[subprocess.Popen] = None
        self.process_start_time: Optional[float] = None
        self.process_temp_dir: str = ""

        # --- GUI-компоненти ---
        self.root = ttk.Window(themename="darkly")
        self.root.title("LinguoAI VideoVoice Pro")
        self.root.geometry("640x850")  # Встановити початковий розмір

        self.source_file = tk.StringVar()
        self.target_file = tk.StringVar()
        self.selected_language = tk.StringVar(value="en")
        self.gemini_key = tk.StringVar()  # Ключ API Gemini
        self.progress_var = tk.DoubleVar(value=0)
        self.log_messages: List[str] = []

        self.languages = {
            "af": ("Африкаанс", "🇿🇦"),
            "sq": ("Албанська", "🇦🇱"),
            "am": ("Амхарська", "🇪🇹"),
            "ar": ("Арабська", "🇸🇦"),
            "eu": ("Баскська", "🇪🇸"),
            "bn": ("Бенгальська", "🇧🇩"),
            "bs": ("Боснійська", "🇧🇦"),
            "bg": ("Болгарська", "🇧🇬"),
            "ca": ("Каталонська", "🇦🇩"),
            "zh-CN": ("Китайська (спрощена)", "🇨🇳"),
            "zh-TW": ("Китайська (традиційна)", "🇹🇼"),
            "hr": ("Хорватська", "🇭🇷"),
            "cs": ("Чеська", "🇨🇿"),
            "da": ("Данська", "🇩🇰"),
            "nl": ("Нідерландська", "🇳🇱"),
            "en": ("Англійська", "🇬🇧"),
            "et": ("Естонська", "🇪🇪"),
            "tl": ("Філіппінська", "🇵🇭"),
            "fi": ("Фінська", "🇫🇮"),
            "fr": ("Французька", "🇫🇷"),
            "gl": ("Галісійська", "🇪🇸"),
            "de": ("Німецька", "🇩🇪"),
            "el": ("Грецька", "🇬🇷"),
            "gu": ("Гуджараті", "🇮🇳"),
            "ha": ("Хауса", "🇳🇬"),
            "he": ("Іврит", "🇮🇱"),
            "hi": ("Гінді", "🇮🇳"),
            "hu": ("Угорська", "🇭🇺"),
            "is": ("Ісландська", "🇮🇸"),
            "id": ("Індонезійська", "🇮🇩"),
            "it": ("Італійська", "🇮🇹"),
            "ja": ("Японська", "🇯🇵"),
            "jw": ("Яванська", "🇮🇩"),
            "kn": ("Каннада", "🇮🇳"),
            "km": ("Кхмерська", "🇰🇭"),
            "ko": ("Корейська", "🇰🇷"),
            "la": ("Латина", "🇻🇦"),
            "lv": ("Латвійська", "🇱🇻"),
            "lt": ("Литовська", "🇱🇹"),
            "ms": ("Малайська", "🇲🇾"),
            "mr": ("Маратхі", "🇮🇳"),
            "ml": ("Малаялам", "🇮🇳"),
            "my": ("М'янма (бірманська)", "🇲🇲"),
            "ne": ("Непальська", "🇳🇵"),
            "no": ("Норвезька", "🇳🇴"),
            "pa": ("Пенджабі", "🇮🇳"),
            "pl": ("Польська", "🇵🇱"),
            "pt": ("Португальська", "🇵🇹"),
            "ro": ("Румунська", "🇷🇴"),
            "ru": ("Російська", "🇷🇺"),
            "sr": ("Сербська", "🇷🇸"),
            "si": ("Сингальська", "🇱🇰"),
            "sk": ("Словацька", "🇸🇰"),
            "sl": ("Словенська", "🇸🇮"),
            "es": ("Іспанська", "🇪🇸"),
            "su": ("Сунданська", "🇮🇩"),
            "sw": ("Суахілі", "🇰🇪"),
            "sv": ("Шведська", "🇸🇪"),
            "ta": ("Тамільська", "🇮🇳"),
            "te": ("Телугу", "🇮🇳"),
            "th": ("Тайська", "🇹🇭"),
            "tr": ("Турецька", "🇹🇷"),
            "uk": ("Українська", "🇺🇦"),
            "ur": ("Урду", "🇵🇰"),
            "vi": ("В'єтнамська", "🇻🇳"),
            "cy": ("Валлійська", "🇬🇧")
        }

        # --- Ініціалізація ---
        self.setup_gui()
        self.setup_ffmpeg()
        self.load_api_keys_from_environment()
        self.load_whisper_model()
        if self.gemini_api_key:
            self.init_gemini()
        self.log_hardware_acceleration()

    def seconds_to_srt_time(self, seconds: float) -> str:
        """Конвертує секунди в формат часу SRT (HH:MM:SS,mmm)."""
        milliseconds = int((seconds * 1000) % 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    def create_srt_file(self, segments: List[Dict], output_path: str):
        """Створює SRT-файл з транскрибованих/перекладених сегментів."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments):
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text']

                    # Конвертуйте секунди в формат часу SRT
                    start_time_srt = self.seconds_to_srt_time(start_time)
                    end_time_srt = self.seconds_to_srt_time(end_time)

                    f.write(f"{i + 1}\n")
                    f.write(f"{start_time_srt} --> {end_time_srt}\n")
                    f.write(f"{text}\n\n")

            self.log_message(f"SRT-файл створено: {output_path}")

        except Exception as e:
            self.log_message(f"Помилка при створенні SRT-файлу: {e}")

    def log_hardware_acceleration(self):
        """Протоколює інформацію про апаратне прискорення."""
        if self.hw_accel_info:
            self.log_message(f"Виявлено апаратне прискорення: {self.hw_accel_info['accel']} ({self.hw_accel_info['info']})")
        else:
            self.log_message("Апаратне прискорення не виявлено.")

    def detect_hardware_acceleration(self):
        """Виявляє апаратне прискорення (NVIDIA, Intel, AMD)."""
        try:
            # NVIDIA
            try:
                subprocess.run(['nvidia-smi'], check=True, capture_output=True)
                return {'accel': 'cuda', 'info': 'Виявлено GPU NVIDIA'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # Intel (Quick Sync)
            try:
                output = subprocess.run(['vainfo'], check=True, capture_output=True, text=True).stdout
                if "VA-API" in output:  # Дуже груба перевірка, можна уточнити
                    return {'accel': 'qsv', 'info': 'Виявлено Intel Quick Sync'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # AMD (AMF)  - трохи складніше, оскільки немає простого "amfinfo"
            #  Можна шукати певні драйвери/пристрої, але це залежить від ОС.
            #  Тут дуже проста, неповна перевірка для Linux:
            if os.name == 'posix':  # Linux/macOS
                try:
                    output = subprocess.run(['lspci', '-v'], check=True, capture_output=True, text=True).stdout
                    if "Advanced Micro Devices, Inc. [AMD/ATI]" in output:
                        return {'accel': 'h264_vaapi', 'info': 'Виявлено GPU AMD (VAAPI)'}  # Припущення!
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass

            return None  # Апаратне прискорення не знайдено

        except Exception as e:
            self.log_message(f"Помилка при виявленні апаратного прискорення: {e}")
            return None
    def init_gemini(self):
        """Ініціалізує модель Gemini Pro."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.log_message("Модель Gemini Pro ініціалізовано.")
        except Exception as e:
            self.log_message(f"Помилка ініціалізації Gemini Pro: {e}")
            messagebox.showerror("Помилка Gemini", f"Не вдалося ініціалізувати Gemini Pro: {e}")
            self.model = None  # Встановити None у разі невдачі

    def check_process_timeout(self):
        """Перевіряє, чи загальний процес перевищив максимально дозволений час."""
        if self.process_start_time and time.time() - self.process_start_time > self.process_timeout:
            if self.current_process:
                self.log_message(f"Таймаут процесу! Завершення процесу (PID: {self.current_process.pid})...")
                self.kill_process_tree(self.current_process.pid)  # Завершити дерево процесів!
            raise TimeoutError("Процес перевищив максимально дозволений час")

    def kill_process_tree(self, pid):
        """Завершує процес та всі його дочірні процеси."""
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):  # Отримати всіх дітей/онуків
                self.log_message(f"Завершення дочірнього процесу: {child.pid}")
                child.kill()
            self.log_message(f"Завершення батьківського процесу: {parent.pid}")
            parent.kill()
        except psutil.NoSuchProcess:
            self.log_message(f"Процес з PID {pid} не знайдено.")
        except Exception as e:
            self.log_message(f"Помилка завершення дерева процесів: {e}")

    def run_subprocess_with_timeout(self, command, timeout, error_message):
        """Запускає підпроцес з динамічним таймаутом та перевірками активності."""
        try:
            self.log_message(f"Виконання команди з таймаутом {timeout}: {' '.join(command)}")
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.log_message(f"Процес запущено з PID: {self.current_process.pid}")

            start_time = time.time()
            last_output_time = start_time

            stdout, stderr = self.current_process.communicate(timeout=timeout)  # Використовуйте communicate!
            retcode = self.current_process.returncode

            if retcode != 0:
                self.log_message(f"Процес завершився з кодом помилки {retcode}:")
                self.log_message(f"Stdout: {stdout}")
                self.log_message(f"Stderr: {stderr}")
                raise subprocess.CalledProcessError(retcode, command, stdout, stderr)

            self.log_message("Команда успішно виконана.")
            return stdout, stderr

        except subprocess.TimeoutExpired:
            self.log_message(f"Таймаут процесу після {timeout} секунд")
            self.kill_process_tree(self.current_process.pid)  # Завершити дерево процесів!
            stdout, stderr = self.current_process.communicate()  # Отримати будь-який залишковий вивід
            self.log_message(f"Stdout: {stdout}")
            self.log_message(f"Stderr: {stderr}")
            raise TimeoutError(error_message)

        except Exception as e:
            self.log_message(f"Виникла неочікувана помилка: {e}")
            if self.current_process:
                self.kill_process_tree(self.current_process.pid)  # Завершити, якщо все ще запущено
            raise
        finally:
            self.current_process = None

    def extract_audio_chunk(self, video_path, audio_path, start_time, duration):
        """Витягує фрагмент аудіо з відео."""
        command = [
            "ffmpeg",
            "-y",  # Перезаписувати вихідні файли без запиту
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-vn",  # Вимкнути відео
            "-acodec", "libmp3lame",
            "-q:a", "2",  # Гарна якість MP3
            "-loglevel", "error",  # Зменшити деталізацію
            audio_path
        ]
        if self.hw_accel:
            command.insert(1, "-hwaccel")
            command.insert(2, self.hw_accel)

        try:
            self.run_subprocess_with_timeout(command, self.chunk_timeout, "Таймаут вилучення аудіо")
            self.log_message(f"Фрагмент аудіо вилучено: '{audio_path}'")
        except Exception as e:
            self.log_message(f"Помилка вилучення фрагмента аудіо: {e}")
            raise

    def batch_segments(self, segments: List[Dict]) -> List[List[Dict]]:
        """Розділяє сегменти на менші пакети для TTS."""
        batched_segments = []
        for i in range(0, len(segments), self.tts_batch_size):
            batch = segments[i:i + self.tts_batch_size]
            batched_segments.append(batch)
        return batched_segments

    def validate_audio_chunk(self, chunk_path: str) -> bool:
        """Перевіряє фрагмент аудіо за допомогою ffprobe."""
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
                f"Таймаут перевірки фрагмента аудіо: {chunk_path}"
            )
            if stdout.strip():
                self.log_message(f"Фрагмент аудіо перевірено: {chunk_path}")
                return True
            else:
                self.log_message(f"Помилка перевірки фрагмента аудіо (немає тривалості): {chunk_path}")
                return False
        except Exception as e:
            self.log_message(f"Помилка перевірки фрагмента аудіо {chunk_path}: {e}")
            return False

    def merge_audio_chunks(self, audio_chunks, output_path):
        """Зливає кілька фрагментів аудіо в один файл за допомогою FFmpeg."""
        if not audio_chunks:
            self.log_message("Немає фрагментів аудіо для злиття.")
            return

        valid_chunks = [chunk for chunk in audio_chunks if self.validate_audio_chunk(chunk)]
        if not valid_chunks:
            self.log_message("Немає дійсних фрагментів аудіо для злиття.")
            return

        timestamp = int(time.time())
        temp_dir = os.path.join(tempfile.gettempdir(), f"audio_merge_temp_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)

        list_file_path = os.path.join(temp_dir, "chunk_list.txt")
        try:
            with open(list_file_path, "w") as f:
                for chunk_path in valid_chunks:
                    abs_chunk_path = os.path.abspath(chunk_path)  # Використовуйте абсолютний шлях
                    f.write(f"file '{abs_chunk_path}'\n")

            command = [
                "ffmpeg",
                "-y",  # Перезаписати
                "-f", "concat",
                "-safe", "0",  # Потрібно для абсолютних шляхів з concat
                "-i", list_file_path,
                "-c", "copy",
                "-loglevel", "error",
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            # Розрахуйте динамічний таймаут на основі кількості фрагментів.
            merge_timeout = len(valid_chunks) * 10 + 30  # 10 секунд на фрагмент + 30 базових
            self.run_subprocess_with_timeout(command, merge_timeout, "Таймаут злиття аудіо")
            self.log_message(f"Фрагменти аудіо злито: '{output_path}'")

        except Exception as e:
            self.log_message(f"Помилка злиття фрагментів аудіо: {e}")
            raise
        finally:
            self.remove_file_with_retry(list_file_path)  # Використовуйте функцію повтору
            self.remove_directory_with_retry(temp_dir)  # і для каталогу

    def merge_video_audio(self, audio_file):
        """Зливає кінцеве аудіо з оригінальним відео."""
        try:
            output_path = self.target_file.get()
            video_path = self.source_file.get()

            # Отримати тривалість відео за допомогою ffprobe
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Не вдалося отримати тривалість відео")
            total_duration = float(duration_output.strip())
            self.log_message(f"Тривалість відео для злиття: {total_duration:.2f} секунд")

            # Розрахуйте динамічний таймаут (наприклад, 3x тривалість + 120 секунд)
            dynamic_timeout = int(3 * total_duration + 120)
            self.log_message(f"Динамічний таймаут для злиття: {dynamic_timeout} секунд")

            command = [
                'ffmpeg',
                '-y',  # Перезаписати вихідний файл
                '-i', video_path,
                '-i', audio_file,
                '-c:v', 'h264_nvenc' if self.hw_accel == 'cuda' else 'libx264',  # H.265
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',  # Завершити кодування, коли закінчується найкоротший потік
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            self.run_subprocess_with_timeout(command, dynamic_timeout, "Таймаут злиття відео та аудіо")
            self.log_message(f"Відео та аудіо злито: {output_path}")
        except Exception as e:
            self.log_message(f"Помилка злиття відео та аудіо: {e}")
            raise

    def shorten_text_with_gemini(self, text: str) -> str:
        """Скорочує текст за допомогою Gemini Pro, обробляючи помилки та обмеження швидкості."""
        if self.model is None:
            self.log_message("Модель Gemini Pro не ініціалізована. Пропускається скорочення.")
            return text

        try:
            prompt = f"Будь ласка, скоротіть наступний текст, зберігаючи ключову інформацію:\n\n{text}"
            time.sleep(1.5)  # Обмеження швидкості: Пауза 1.5 секунди
            response = self.model.generate_content(prompt)
            if response and response.text:
                shortened_text = response.text
                self.log_message("Текст скорочено за допомогою Gemini Pro.")
                return shortened_text
            else:
                self.log_message("Gemini Pro повернув порожню відповідь. Використовується оригінальний текст.")
                return text
        except Exception as e:
            self.log_message(f"Помилка скорочення тексту за допомогою Gemini: {e}")
            return text

    def process_video(self):
        """Основний робочий процес обробки відео."""
        self.process_start_time = time.time()
        timestamp = int(time.time())
        self.process_temp_dir = os.path.join(tempfile.gettempdir(), f"process_temp_{timestamp}")
        os.makedirs(self.process_temp_dir, exist_ok=True)

        try:
            self.log_message("Розпочинається обробка відео...")
            self.progress_var.set(5)

            # Налаштуйте перевірку таймауту *перед* початком будь-яких завдань.
            def check_timeout():
                self.check_process_timeout()
                self.root.after(1000, check_timeout)  # Перевірка кожну секунду

            check_timeout()  # Запустити перевірку таймауту

            self.progress_var.set(10)
            audio_chunks, total_duration = self.extract_audio_in_chunks(self.source_file.get())
            self.progress_var.set(25)

            segments, total_words_original = self.transcribe_audio_in_chunks(audio_chunks, self.target_language)
            if not segments:
                raise Exception("Транскрипція не вдалася.")
            self.progress_var.set(45)

            translated_segments, total_words_translated = self.translate_and_refine_segments(segments)

            # --- Скорочення тексту Gemini (Необов'язково) ---
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
                    self.root.update()  # Оновити GUI
                translated_segments = shortened_segments

                shortening_change = (
                    ((total_words_translated - total_words_shortened) / total_words_translated) * 100
                    if total_words_translated else 0
                )
                self.log_message(f"Скорочення зменшило кількість слів на: {shortening_change:.2f}%")
            else:
                total_words_shortened = total_words_translated
                self.log_message("Ключ API Gemini не надано. Пропускається скорочення.")

            self.progress_var.set(60)

            # Створення SRT-файлу (приклад)
            srt_output_path = os.path.splitext(self.target_file.get())[0] + ".srt"  # Відповідно до назви вихідного відео
            self.create_srt_file(translated_segments, srt_output_path)

            # --- Звіт про кількість слів ---
            if total_words_original > 0:
                translation_change = (total_words_translated / total_words_original) * 100
                self.log_message(f"Початкова кількість слів: {total_words_original}")
                self.log_message(f"Перекладена кількість слів: {total_words_translated}")
                self.log_message(f"Зміна кількості слів після перекладу: {translation_change:.2f}%")
            else:
                self.log_message("Початкова кількість слів дорівнює нулю. Пропускається відсоток.")

            # --- TTS та злиття аудіо ---
            batched_translated_segments = self.batch_segments(translated_segments)
            all_updated_segments = []
            merged_audio_files = []

            for i, batch in enumerate(batched_translated_segments):
                self.log_message(f"Обробка пакета TTS {i+1}/{len(batched_translated_segments)}")
                merged_audio_path, updated_segments = self.generate_tts_audio_for_segments(batch)
                if merged_audio_path:
                    all_updated_segments.extend(updated_segments)
                    merged_audio_files.append(merged_audio_path)
                else:
                    self.log_message(f"Пакет TTS {i+1} не вдалося.")
                    # Очистити *всі* раніше створені файли TTS у разі невдачі
                    for file_path in merged_audio_files:
                        self.remove_file_with_retry(file_path)
                    raise Exception(f"Не вдалося створити аудіо TTS для пакета {i + 1}.")

            # --- Кінцеве злиття аудіо (якщо кілька пакетів) ---
            if len(merged_audio_files) > 1:
                final_merged_audio_path = os.path.join(self.process_temp_dir,
                                                        f"final_merged_tts_audio_{timestamp}.mp3")
                self.merge_audio_chunks(merged_audio_files, final_merged_audio_path)
                self.log_message(f"Кінцеве злите аудіо TTS: {final_merged_audio_path}")
            elif merged_audio_files:
                final_merged_audio_path = merged_audio_files[0]
                self.log_message(f"Кінцеве злите аудіо TTS (один пакет): {final_merged_audio_path}")
            else:
                raise Exception("Не створено аудіо TTS.")

            # --- Очистити проміжні файли TTS ---
            for file_path in merged_audio_files:
                if file_path != final_merged_audio_path:  # Не видаляйте кінцевий файл!
                    self.remove_file_with_retry(file_path)

            self.progress_var.set(80)

            # --- Кінцеве злиття відео/аудіо ---
            self.merge_video_audio(final_merged_audio_path)
            self.progress_var.set(95)

            self.log_message("Обробку успішно завершено! 🎉")
            messagebox.showinfo("Успіх", "Обробку відео завершено!")


        except TimeoutError as e:
            self.log_message(f"Таймаут обробки: {str(e)}")
            messagebox.showerror("Помилка", f"Таймаут обробки: {str(e)}")
        except Exception as e:
            self.log_message(f"Помилка: {str(e)}")
            messagebox.showerror("Помилка", f"Обробка не вдалася: {str(e)}")
        finally:
            # --- Очищення ---
            self.progress_var.set(0)
            self.process_start_time = None
            self.current_process = None
            self.start_button.config(state=tk.NORMAL)  # Знову ввімкнути кнопку
            self.remove_directory_with_retry(self.process_temp_dir)

    def remove_file_with_retry(self, file_path, retries=3, delay=0.5):
        """Видаляє файл, повторюючи спроби за потреби."""
        file_path = os.path.abspath(file_path)  # Використовуйте абсолютний шлях
        for i in range(retries):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                self.log_message(f"Файл видалено: {file_path}")
                return  # Успіх
            except OSError as e:
                if e.errno == errno.ENOENT:  # Файл не знайдено - вже видалено
                    self.log_message(f"Файл не знайдено (вже видалено): {file_path}")
                    return
                if i < retries - 1:  # Не чекайте на останню спробу
                    self.log_message(f"Повторна спроба видалення файлу ({i+1}/{retries}): {file_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Помилка видалення файлу після кількох повторних спроб: {file_path} - {e}")
            except Exception as e:
                self.log_message(f"Неочікувана помилка видалення файлу: {file_path} - {e}")
                return  # Не повторюйте спроби для неочікуваних помилок

    def remove_directory_with_retry(self, dir_path, retries=5, delay=1):
        """Видаляє каталог, повторюючи спроби за потреби (особливо для непустих)."""
        dir_path = os.path.abspath(dir_path)
        for i in range(retries):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                self.log_message(f"Каталог видалено: {dir_path}")
                return
            except OSError as e:
                if e.errno == errno.ENOENT:  # Каталог вже видалено
                    self.log_message(f"Каталог вже видалено: {dir_path}")
                    return
                elif e.errno == errno.ENOTEMPTY:  # Каталог не пустий
                    self.log_message(f"Каталог не пустий, повторна спроба видалення ({i+1}/{retries}): {dir_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Помилка видалення каталогу: {dir_path} - {e}")
                    time.sleep(delay)  # Чекайте навіть на інші помилки
            except Exception as e:
                self.log_message(f"Неочікувана помилка видалення каталогу: {dir_path} - {e}")
                return

    def setup_ffmpeg(self):
        """Перевіряє, чи доступний FFmpeg."""
        try:
            self.run_subprocess_with_timeout(['ffmpeg', '-version'], 10, "Перевірка FFmpeg не вдалася")
            self.ffmpeg_available = True
            self.log_message("FFmpeg виявлено.")
        except FileNotFoundError:
            self.ffmpeg_available = False
            self.log_message("FFmpeg не знайдено. Встановіть FFmpeg.")
            messagebox.showwarning("FFmpeg відсутній", "Потрібен FFmpeg. Будь ласка, встановіть його.")
        except Exception as e:
            self.ffmpeg_available = False
            self.log_message(f"Перевірка FFmpeg не вдалася: {e}")
            messagebox.showwarning("Помилка FFmpeg", "Перевірка FFmpeg не вдалася. Перевірте встановлення.")

    def setup_gui(self):
        """Налаштовує графічний інтерфейс користувача."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Заголовок ---
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        title_label = ttk.Label(header_frame, text="🎬 LinguoAI VideoVoice Pro", font=('Helvetica', 16, 'bold'))
        title_label.pack()

        # --- Вибір файлів ---
        file_frame = ttk.LabelFrame(main_frame, text="Відеофайли", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        # Використовуйте сіткове розташування для рядків вводу/виводу
        source_frame = ttk.Frame(file_frame)
        source_frame.pack(fill=tk.X, pady=2)
        ttk.Label(source_frame, text="📹 Вхідний:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_file, width=40)
        self.source_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(source_frame, text="Огляд", command=self.browse_source, style="Accent.TButton").grid(row=0, column=2, padx=5)

        target_frame = ttk.Frame(file_frame)
        target_frame.pack(fill=tk.X, pady=2)
        ttk.Label(target_frame, text="💾 Вихідний:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.target_entry = ttk.Entry(target_frame, textvariable=self.target_file, width=40)
        self.target_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(target_frame, text="Огляд", command=self.browse_target, style="Accent.TButton").grid(row=0, column=2, padx=5)

        # Зробіть стовпці введення розширюваними
        source_frame.columnconfigure(1, weight=1)
        target_frame.columnconfigure(1, weight=1)


        # --- Вибір мови ---
        lang_frame = ttk.LabelFrame(main_frame, text="Налаштування голосу", padding="10")
        lang_frame.pack(fill=tk.X, pady=5)

        lang_combo_frame = ttk.Frame(lang_frame)
        lang_combo_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lang_combo_frame, text="🗣️ Цільова мова:").pack(side=tk.LEFT, padx=5)

        # Пошуковий Combobox
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
        self.lang_combo.set("🇬🇧 Англійська")  # Встановити значення за замовчуванням після створення combobox
        self.language_search_entry.bind("<Return>", (lambda event: self.lang_combo.focus()))
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_language_selected)

        # --- Параметри злиття ---
        options_frame = ttk.LabelFrame(main_frame, text="Параметри злиття", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        # Ключ API Gemini
        brain_frame = ttk.Frame(options_frame)
        brain_frame.pack(fill=tk.X, pady=2)

        # --- Опис ключа API Gemini ---
        gemini_description_label = ttk.Label(
            brain_frame,
            wraplength=600,  # Переносити текст
            justify=tk.LEFT,  # Вирівнювання тексту по лівому краю
            text="Ця програма використовує Google Gemini Pro API для необов'язкового скорочення тексту.  "
                 "Це може допомогти зменшити загальну довжину перекладеного тексту, зберігаючи при цьому ключову інформацію.\n"
                 "Для використання цієї функції потрібен ключ API.  Якщо у вас немає ключа, ви можете пропустити цей крок, "
                 "і програма продовжить роботу без скорочення."
        )
        gemini_description_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        gemini_link_label = ttk.Label(
            brain_frame,
            text="Отримати ключ API Gemini тут: ",
            foreground="dodgerblue",  # Змінено на більш тонкий синій
            cursor="hand2"  # Змінити курсор при наведенні
        )
        gemini_link_label.pack(side=tk.TOP, fill=tk.X)
        gemini_link_label.bind("<Button-1>", lambda e: self.open_webpage("https://makersuite.google.com/app/apikey"))

        self.gemini_key_label = ttk.Label(brain_frame, text="Ключ API Gemini:")
        self.gemini_key_label.pack(side=tk.LEFT, padx=5)
        self.gemini_key_entry = ttk.Entry(brain_frame, textvariable=self.gemini_key, width=30, show="*")
        self.gemini_key_entry.pack(side=tk.LEFT)

        self.start_button = ttk.Button(options_frame, text="🚀 Розпочати обробку", command=self.start_processing,
                                       style="Success.TButton")
        self.start_button.pack(pady=10)

        # --- Прогрес ---
        progress_frame = ttk.LabelFrame(main_frame, text="Прогрес", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100,
                                            style="Success.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(progress_frame, text="Готовий")
        self.status_label.pack()

        # --- Журнал ---
        log_frame = ttk.LabelFrame(main_frame, text="Журнал обробки", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, state='disabled', wrap=tk.WORD)  # Увімкнути перенесення слів
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def on_language_selected(self, event):
        """Обробляє вибір мови з combobox."""
        selected_lang_str = self.lang_combo.get()
        for code, (name, flag) in self.languages.items():
            if f"{flag} {name}" == selected_lang_str:
                self.target_language = code  # Оновити цільову мову
                self.log_message(f"Вибрано цільову мову: {name} ({code})")
                break

    def update_language_list(self, *args):
        """Фільтрує список мов на основі пошукового запиту."""
        search_term = self.language_search_var.get().lower()
        filtered_languages = [
            f"{flag} {name}"
            for code, (name, flag) in self.languages.items()
            if search_term in name.lower() or search_term in code.lower() or search_term in flag.lower()
        ]
        self.lang_combo['values'] = filtered_languages
        if filtered_languages:
            self.lang_combo.set(filtered_languages[0])  # Встановити першу відповідність
        else:
            self.lang_combo.set('')  # Очистити, якщо немає відповідностей
    def browse_source(self):
        """Відкриває діалогове вікно вибору файлу для вибору вихідного відео."""
        filename = filedialog.askopenfilename(
            title="Виберіть відеофайл",
            filetypes=[("Відеофайли", "*.mp4 *.avi *.mkv *.mov"), ("Усі файли", "*.*")]
        )
        if filename:
            self.source_file.set(filename)
            base = os.path.splitext(filename)[0]
            self.target_file.set(f"{base}_translated.mp4")
            self.source_entry.xview_moveto(1) #Прокрутити до кінця
            self.target_entry.xview_moveto(1)

    def browse_target(self):
        """Відкриває діалогове вікно збереження файлу для вибору шляху до вихідного відео."""
        filename = filedialog.asksaveasfilename(
            title="Зберегти перекладене відео",
            defaultextension=".mp4",
            filetypes=[("MP4 файли", "*.mp4")]
        )
        if filename:
            self.target_file.set(filename)
            self.source_entry.xview_moveto(1)  # Прокрутити до кінця
            self.target_entry.xview_moveto(1)
    def log_message(self, message):
        """Записує повідомлення до GUI та внутрішнього списку журналу."""
        self.log_messages.append(message)
        self.log_text.config(state='normal')  # Тимчасово увімкнути
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')  # Знову вимкнути
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def load_api_keys_from_environment(self):
        """Завантажує ключі API зі змінних середовища."""
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_key.set(self.gemini_api_key)  # Завантажити в GUI

    def start_processing(self):
        """Запускає обробку відео в окремому потоці."""
        if not self.ffmpeg_available:
            messagebox.showerror("Помилка", "Потрібен FFmpeg!")
            return
        if not self.source_file.get() or not self.target_file.get():
            messagebox.showerror("Помилка", "Виберіть вхідний та вихідний файли.")
            return

        try:
            lang_str = self.lang_combo.get()
            self.target_language = next(
                (code for code, (name, flag) in self.languages.items() if f"{flag} {name}" == lang_str), None
            )
            if self.target_language is None:
                raise ValueError("Вибрано недійсну мову.")
        except ValueError as e:
            messagebox.showerror("Помилка", str(e))
            return

        self.gemini_api_key = self.gemini_key.get()
        if self.gemini_api_key:
            self.init_gemini()
        else:
            self.log_message("Ключ API Gemini не надано. Пропускається скорочення.")

        self.start_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.log_text.config(state='normal')
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state='disabled')
        self.log_messages = []
        threading.Thread(target=self.process_video, daemon=True).start()

    def load_whisper_model(self):
        """Завантажує модель Faster Whisper."""
        try:
            self.log_message(f"Завантаження моделі Whisper ({self.model_size}) на {self.device}...")
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.log_message("Модель Whisper успішно завантажено.")
        except Exception as e:
            self.log_message(f"Помилка завантаження моделі Whisper: {e}")
            raise

    def extract_audio_in_chunks(self, video_path: str) -> Tuple[List[str], float]:
        """Витягує аудіо з відео фрагментами."""
        self.log_message(f"Витягування аудіо з: {video_path}")
        audio_chunks = []
        try:
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Не вдалося отримати тривалість відео")
            total_duration = float(duration_output.strip())
            self.log_message(f"Загальна тривалість відео: {total_duration:.2f} секунд")

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
            self.log_message(f"Помилка під час вилучення аудіо: {e}")
            raise  # Повторно викинути виняток для обробки в process_video

    def transcribe_audio_in_chunks(self, audio_chunks: List[str], language: str) -> Tuple[List[Dict], int]:
        """Транскрибує фрагменти аудіо за допомогою Faster Whisper."""
        all_segments = []
        total_words = 0
        if self.whisper_model is None:
            raise ValueError("Модель Whisper не завантажено.")

        try:
            for i, chunk_path in enumerate(audio_chunks):
                self.log_message(f"Транскрибування фрагмента {i+1}/{len(audio_chunks)}: {chunk_path}")
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
                self.log_message(f"Транскрипцію фрагмента {i+1} завершено.")
                self.root.update()  # Оновити GUI

                self.remove_file_with_retry(chunk_path)
                self.remove_directory_with_retry(os.path.dirname(chunk_path))

            self.log_message(f"Загальна кількість транскрибованих слів: {total_words}")
            return all_segments, total_words
        except Exception as e:
            self.log_message(f"Помилка під час транскрипції: {e}")
            raise

    def translate_and_refine_segments(self, segments: List[Dict]) -> Tuple[List[Dict], int]:
        """Перекладає сегменти та обробляє можливі помилки перекладу."""
        translated_segments = []
        total_words_translated = 0
        translator = GoogleTranslator(source='auto', target=self.target_language)

        try:
            for segment in segments:
                self.log_message(f"Переклад сегмента: '{segment['text'][:50]}...' на {self.target_language}")
                translated_text = translator.translate(segment['text'])
                if not isinstance(translated_text, str):
                    self.log_message(
                        f"ПОПЕРЕДЖЕННЯ: translator.translate не повернув рядок. Тип: {type(translated_text)}, Значення: {translated_text}")
                    translated_text = ""  # Встановити порожній рядок
                translated_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': translated_text
                })
                total_words_translated += len(translated_text.split())
                self.log_message(f"Перекладений сегмент: '{translated_text[:50]}...'")
            return translated_segments, total_words_translated
        except Exception as e:
            self.log_message(f"Помилка під час перекладу: {e}")
            raise

    def generate_tts_audio_for_segments(self, translated_segments: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
        """Створює аудіо TTS для пакета перекладених сегментів."""
        tts_chunks = []
        updated_segments = []
        total_duration = 0
        try:
            for i, segment in enumerate(translated_segments):
                text = segment['text']
                self.log_message(f"Створення TTS для сегмента {i+1}/{len(translated_segments)}: '{text[:50]}...'")
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
                                                                        "Не вдалося отримати тривалість сегмента TTS")
                    segment_duration = float(duration_output.strip())
                    updated_segments.append({
                        'start': total_duration,
                        'end': total_duration + segment_duration,
                        'text': text,
                    })
                    total_duration += segment_duration
                except Exception as e:
                    self.log_message(f"Помилка створення TTS для сегмента {i + 1}: {e}")
                    # Очистити будь-які створені файли в цьому пакеті до цього часу
                    for file in tts_chunks:
                        self.remove_file_with_retry(file)
                        self.remove_directory_with_retry(os.path.dirname(file))  # Спробуйте видалити каталог
                    return None, []  # Вказати на невдачу

                self.log_message(f"TTS для сегмента {i+1} створено.")
                self.root.update()  # Підтримувати чутливість GUI

            self.log_message(f"Злиття {len(tts_chunks)} фрагментів TTS...")
            if tts_chunks:
                merged_audio_path = os.path.join(self.process_temp_dir, f"merged_tts_audio_{int(time.time())}.mp3")
                self.merge_audio_chunks(tts_chunks, merged_audio_path)
                self.log_message(f"Злите аудіо TTS: {merged_audio_path}")

                for tts_file in tts_chunks:
                    self.remove_file_with_retry(tts_file)
                    self.remove_directory_with_retry(os.path.dirname(tts_file))

                return merged_audio_path, updated_segments
            else:
                self.log_message("Не створено сегментів TTS.")
                return None, []

        except Exception as e:
            self.log_message(f"Помилка під час створення/злиття TTS: {e}")
            for file in tts_chunks:  # Виправлено
                self.remove_file_with_retry(file)
                self.remove_directory_with_retry(os.path.dirname(file))  # і каталог
            return None, []

    def open_webpage(self, url):
        """Відкриває веб-сторінку у браузері за замовчуванням."""
        import webbrowser
        webbrowser.open(url)

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Завантаження токенізатора NLTK punkt...")
        nltk.download('punkt')
    app = LinguoAIVideoVoicePro()
    app.root.mainloop()