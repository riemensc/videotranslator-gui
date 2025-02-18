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
        # --- Параметры конфигурации ---
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
        self.hw_accel_info = self.detect_hardware_acceleration()  # Сохраняем информацию
        self.hw_accel = self.hw_accel_info['accel'] if self.hw_accel_info else None

        # --- Внутреннее состояние ---
        self.whisper_model: Optional[WhisperModel] = None
        self.current_process: Optional[subprocess.Popen] = None
        self.process_start_time: Optional[float] = None
        self.process_temp_dir: str = ""

        # --- GUI-компоненты ---
        self.root = ttk.Window(themename="darkly")
        self.root.title("LinguoAI VideoVoice Pro")
        self.root.geometry("640x850")  # Устанавливаем начальный размер

        self.source_file = tk.StringVar()
        self.target_file = tk.StringVar()
        self.selected_language = tk.StringVar(value="en")
        self.gemini_key = tk.StringVar()  # Ключ API Gemini
        self.progress_var = tk.DoubleVar(value=0)
        self.log_messages: List[str] = []

        self.languages = {
            "af": ("Африкаанс", "🇿🇦"),
            "sq": ("Албанский", "🇦🇱"),
            "am": ("Амхарский", "🇪🇹"),
            "ar": ("Арабский", "🇸🇦"),
            "eu": ("Баскский", "🇪🇸"),
            "bn": ("Бенгальский", "🇧🇩"),
            "bs": ("Боснийский", "🇧🇦"),
            "bg": ("Болгарский", "🇧🇬"),
            "ca": ("Каталанский", "🇦🇩"),
            "zh-CN": ("Китайский (Упрощенный)", "🇨🇳"),
            "zh-TW": ("Китайский (Традиционный)", "🇹🇼"),
            "hr": ("Хорватский", "🇭🇷"),
            "cs": ("Чешский", "🇨🇿"),
            "da": ("Датский", "🇩🇰"),
            "nl": ("Голландский", "🇳🇱"),
            "en": ("Английский", "🇬🇧"),
            "et": ("Эстонский", "🇪🇪"),
            "tl": ("Филиппинский", "🇵🇭"),
            "fi": ("Финский", "🇫🇮"),
            "fr": ("Французский", "🇫🇷"),
            "gl": ("Галисийский", "🇪🇸"),
            "de": ("Немецкий", "🇩🇪"),
            "el": ("Греческий", "🇬🇷"),
            "gu": ("Гуджарати", "🇮🇳"),
            "ha": ("Хауса", "🇳🇬"),
            "he": ("Иврит", "🇮🇱"),
            "hi": ("Хинди", "🇮🇳"),
            "hu": ("Венгерский", "🇭🇺"),
            "is": ("Исландский", "🇮🇸"),
            "id": ("Индонезийский", "🇮🇩"),
            "it": ("Итальянский", "🇮🇹"),
            "ja": ("Японский", "🇯🇵"),
            "jw": ("Яванский", "🇮🇩"),
            "kn": ("Каннада", "🇮🇳"),
            "km": ("Кхмерский", "🇰🇭"),
            "ko": ("Корейский", "🇰🇷"),
            "la": ("Латинский", "🇻🇦"),
            "lv": ("Латышский", "🇱🇻"),
            "lt": ("Литовский", "🇱🇹"),
            "ms": ("Малайский", "🇲🇾"),
            "mr": ("Маратхи", "🇮🇳"),
            "ml": ("Малаялам", "🇮🇳"),
            "my": ("Мьянманский (Бирманский)", "🇲🇲"),
            "ne": ("Непальский", "🇳🇵"),
            "no": ("Норвежский", "🇳🇴"),
            "pa": ("Панджаби", "🇮🇳"),
            "pl": ("Польский", "🇵🇱"),
            "pt": ("Португальский", "🇵🇹"),
            "ro": ("Румынский", "🇷🇴"),
            "ru": ("Русский", "🇷🇺"),
            "sr": ("Сербский", "🇷🇸"),
            "si": ("Сингальский", "🇱🇰"),
            "sk": ("Словацкий", "🇸🇰"),
            "sl": ("Словенский", "🇸🇮"),
            "es": ("Испанский", "🇪🇸"),
            "su": ("Сунданский", "🇮🇩"),
            "sw": ("Суахили", "🇰🇪"),
            "sv": ("Шведский", "🇸🇪"),
            "ta": ("Тамильский", "🇮🇳"),
            "te": ("Телугу", "🇮🇳"),
            "th": ("Тайский", "🇹🇭"),
            "tr": ("Турецкий", "🇹🇷"),
            "uk": ("Украинский", "🇺🇦"),
            "ur": ("Урду", "🇵🇰"),
            "vi": ("Вьетнамский", "🇻🇳"),
            "cy": ("Валлийский", "🇬🇧")
        }

        # --- Инициализация ---
        self.setup_gui()
        self.setup_ffmpeg()
        self.load_api_keys_from_environment()
        self.load_whisper_model()
        if self.gemini_api_key:
            self.init_gemini()
        self.log_hardware_acceleration()

    def seconds_to_srt_time(self, seconds: float) -> str:
        """Преобразует секунды в формат времени SRT (HH:MM:SS,mmm)."""
        milliseconds = int((seconds * 1000) % 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    def create_srt_file(self, segments: List[Dict], output_path: str):
        """Создает SRT-файл из транскрибированных/переведенных сегментов."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments):
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text']

                    # Преобразование секунд в формат SRT
                    start_time_srt = self.seconds_to_srt_time(start_time)
                    end_time_srt = self.seconds_to_srt_time(end_time)

                    f.write(f"{i + 1}\n")
                    f.write(f"{start_time_srt} --> {end_time_srt}\n")
                    f.write(f"{text}\n\n")

            self.log_message(f"SRT-файл создан: {output_path}")

        except Exception as e:
            self.log_message(f"Ошибка при создании SRT-файла: {e}")

    def log_hardware_acceleration(self):
        """Записывает информацию об аппаратном ускорении."""
        if self.hw_accel_info:
            self.log_message(f"Обнаружено аппаратное ускорение: {self.hw_accel_info['accel']} ({self.hw_accel_info['info']})")
        else:
            self.log_message("Аппаратное ускорение не обнаружено.")

    def detect_hardware_acceleration(self):
        """Обнаруживает аппаратное ускорение (NVIDIA, Intel, AMD)."""
        try:
            # NVIDIA
            try:
                subprocess.run(['nvidia-smi'], check=True, capture_output=True)
                return {'accel': 'cuda', 'info': 'Обнаружен GPU NVIDIA'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # Intel (Quick Sync)
            try:
                output = subprocess.run(['vainfo'], check=True, capture_output=True, text=True).stdout
                if "VA-API" in output:  # Очень грубая проверка, можно улучшить
                    return {'accel': 'qsv', 'info': 'Обнаружен Intel Quick Sync'}
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # AMD (AMF) - немного сложнее, так как нет простого "amfinfo"
            # Можно было бы поискать определенные драйверы/устройства, но это зависит от ОС.
            # Здесь простая, неполная проверка для Linux:
            if os.name == 'posix':  # Linux/macOS
                try:
                    output = subprocess.run(['lspci', '-v'], check=True, capture_output=True, text=True).stdout
                    if "Advanced Micro Devices, Inc. [AMD/ATI]" in output:
                        return {'accel': 'h264_vaapi', 'info': 'Обнаружен GPU AMD (VAAPI)'}  # Предположение!
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass

            return None  # Аппаратное ускорение не найдено

        except Exception as e:
            self.log_message(f"Ошибка при обнаружении аппаратного ускорения: {e}")
            return None
    def init_gemini(self):
        """Инициализирует модель Gemini Pro."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.log_message("Модель Gemini Pro инициализирована.")
        except Exception as e:
            self.log_message(f"Ошибка при инициализации Gemini Pro: {e}")
            messagebox.showerror("Gemini Error", f"Не удалось инициализировать Gemini Pro: {e}")
            self.model = None  # Устанавливаем None при неудаче

    def check_process_timeout(self):
        """Проверяет, не превысило ли время выполнения процесса максимально допустимое."""
        if self.process_start_time and time.time() - self.process_start_time > self.process_timeout:
            if self.current_process:
                self.log_message(f"Превышено время ожидания процесса! Завершаем процесс (PID: {self.current_process.pid})...")
                self.kill_process_tree(self.current_process.pid)  # Завершаем дерево процессов!
            raise TimeoutError("Время выполнения процесса превысило максимально допустимое")

    def kill_process_tree(self, pid):
        """Завершает процесс и все его дочерние процессы."""
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):  # Получаем всех детей/внуков
                self.log_message(f"Завершаем дочерний процесс: {child.pid}")
                child.kill()
            self.log_message(f"Завершаем родительский процесс: {parent.pid}")
            parent.kill()
        except psutil.NoSuchProcess:
            self.log_message(f"Процесс с PID {pid} не найден.")
        except Exception as e:
            self.log_message(f"Ошибка при завершении дерева процессов: {e}")

    def run_subprocess_with_timeout(self, command, timeout, error_message):
        """Запускает подпроцесс с динамическим тайм-аутом и проверками активности."""
        try:
            self.log_message(f"Запуск команды с тайм-аутом {timeout}: {' '.join(command)}")
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.log_message(f"Процесс запущен с PID: {self.current_process.pid}")

            start_time = time.time()
            last_output_time = start_time

            stdout, stderr = self.current_process.communicate(timeout=timeout)  # Используем communicate!
            retcode = self.current_process.returncode

            if retcode != 0:
                self.log_message(f"Процесс завершился с кодом ошибки {retcode}:")
                self.log_message(f"Stdout: {stdout}")
                self.log_message(f"Stderr: {stderr}")
                raise subprocess.CalledProcessError(retcode, command, stdout, stderr)

            self.log_message("Команда успешно выполнена.")
            return stdout, stderr

        except subprocess.TimeoutExpired:
            self.log_message(f"Превышено время ожидания процесса после {timeout} секунд")
            self.kill_process_tree(self.current_process.pid)  # Завершаем дерево процессов!
            stdout, stderr = self.current_process.communicate()  # Получаем оставшийся вывод
            self.log_message(f"Stdout: {stdout}")
            self.log_message(f"Stderr: {stderr}")
            raise TimeoutError(error_message)

        except Exception as e:
            self.log_message(f"Произошла непредвиденная ошибка: {e}")
            if self.current_process:
                self.kill_process_tree(self.current_process.pid)  # Завершаем, если еще выполняется
            raise
        finally:
            self.current_process = None

    def extract_audio_chunk(self, video_path, audio_path, start_time, duration):
        """Извлекает фрагмент аудио из видео."""
        command = [
            "ffmpeg",
            "-y",  # Перезаписываем выходные файлы без запроса
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-vn",  # Отключаем видео
            "-acodec", "libmp3lame",
            "-q:a", "2",  # Хорошее качество MP3
            "-loglevel", "error",  # Уменьшаем детализацию
            audio_path
        ]
        if self.hw_accel:
            command.insert(1, "-hwaccel")
            command.insert(2, self.hw_accel)

        try:
            self.run_subprocess_with_timeout(command, self.chunk_timeout, "Превышено время ожидания извлечения аудио")
            self.log_message(f"Аудиофрагмент извлечен: '{audio_path}'")
        except Exception as e:
            self.log_message(f"Ошибка при извлечении аудиофрагмента: {e}")
            raise

    def batch_segments(self, segments: List[Dict]) -> List[List[Dict]]:
        """Разбивает сегменты на небольшие пакеты для TTS."""
        batched_segments = []
        for i in range(0, len(segments), self.tts_batch_size):
            batch = segments[i:i + self.tts_batch_size]
            batched_segments.append(batch)
        return batched_segments

    def validate_audio_chunk(self, chunk_path: str) -> bool:
        """Проверяет аудиофрагмент с использованием ffprobe."""
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
                f"Превышено время ожидания проверки аудиофрагмента: {chunk_path}"
            )
            if stdout.strip():
                self.log_message(f"Аудиофрагмент проверен: {chunk_path}")
                return True
            else:
                self.log_message(f"Проверка аудиофрагмента не удалась (нет длительности): {chunk_path}")
                return False
        except Exception as e:
            self.log_message(f"Ошибка при проверке аудиофрагмента {chunk_path}: {e}")
            return False

    def merge_audio_chunks(self, audio_chunks, output_path):
        """Объединяет несколько аудиофрагментов в один файл с использованием FFmpeg."""
        if not audio_chunks:
            self.log_message("Нет аудиофрагментов для объединения.")
            return

        valid_chunks = [chunk for chunk in audio_chunks if self.validate_audio_chunk(chunk)]
        if not valid_chunks:
            self.log_message("Нет допустимых аудиофрагментов для объединения.")
            return

        timestamp = int(time.time())
        temp_dir = os.path.join(tempfile.gettempdir(), f"audio_merge_temp_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)

        list_file_path = os.path.join(temp_dir, "chunk_list.txt")
        try:
            with open(list_file_path, "w") as f:
                for chunk_path in valid_chunks:
                    abs_chunk_path = os.path.abspath(chunk_path)  # Используем абсолютный путь
                    f.write(f"file '{abs_chunk_path}'\n")

            command = [
                "ffmpeg",
                "-y",  # Перезаписываем
                "-f", "concat",
                "-safe", "0",  # Требуется для абсолютных путей с concat
                "-i", list_file_path,
                "-c", "copy",
                "-loglevel", "error",
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            # Рассчитываем динамический тайм-аут на основе количества фрагментов.
            merge_timeout = len(valid_chunks) * 10 + 30  # 10 секунд на фрагмент + 30 базовых
            self.run_subprocess_with_timeout(command, merge_timeout, "Превышено время ожидания объединения аудио")
            self.log_message(f"Аудиофрагменты объединены: '{output_path}'")

        except Exception as e:
            self.log_message(f"Ошибка при объединении аудиофрагментов: {e}")
            raise
        finally:
            self.remove_file_with_retry(list_file_path)  # Используем функцию повтора
            self.remove_directory_with_retry(temp_dir)  # и для каталога

    def merge_video_audio(self, audio_file):
        """Объединяет финальный звук с исходным видео."""
        try:
            output_path = self.target_file.get()
            video_path = self.source_file.get()

            # Получаем длительность видео с помощью ffprobe
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Не удалось получить длительность видео")
            total_duration = float(duration_output.strip())
            self.log_message(f"Длительность видео для объединения: {total_duration:.2f} секунд")

            # Рассчитываем динамический тайм-аут (например, 3x длительность + 120 секунд)
            dynamic_timeout = int(3 * total_duration + 120)
            self.log_message(f"Динамический тайм-аут для объединения: {dynamic_timeout} секунд")

            command = [
                'ffmpeg',
                '-y',  # Перезаписываем выходной файл
                '-i', video_path,
                '-i', audio_file,
                '-c:v', 'h264_nvenc' if self.hw_accel == 'cuda' else 'libx264',  # H.265
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',  # Завершаем кодирование, когда заканчивается самый короткий поток
                output_path
            ]
            if self.hw_accel:
                command.insert(1, "-hwaccel")
                command.insert(2, self.hw_accel)

            self.run_subprocess_with_timeout(command, dynamic_timeout, "Превышено время ожидания объединения видео и аудио")
            self.log_message(f"Видео и аудио объединены: {output_path}")
        except Exception as e:
            self.log_message(f"Ошибка при объединении видео и аудио: {e}")
            raise

    def shorten_text_with_gemini(self, text: str) -> str:
        """Сокращает текст с помощью Gemini Pro, обрабатывая ошибки и ограничения скорости."""
        if self.model is None:
            self.log_message("Модель Gemini Pro не инициализирована. Пропускаем сокращение.")
            return text

        try:
            prompt = f"Пожалуйста, сократите следующий текст, сохранив ключевую информацию:\n\n{text}"
            time.sleep(1.5)  # Ограничение скорости: пауза на 1,5 секунды
            response = self.model.generate_content(prompt)
            if response and response.text:
                shortened_text = response.text
                self.log_message("Текст сокращен с помощью Gemini Pro.")
                return shortened_text
            else:
                self.log_message("Gemini Pro вернул пустой ответ. Используем исходный текст.")
                return text
        except Exception as e:
            self.log_message(f"Ошибка при сокращении текста с помощью Gemini: {e}")
            return text

    def process_video(self):
        """Основной рабочий процесс обработки видео."""
        self.process_start_time = time.time()
        timestamp = int(time.time())
        self.process_temp_dir = os.path.join(tempfile.gettempdir(), f"process_temp_{timestamp}")
        os.makedirs(self.process_temp_dir, exist_ok=True)

        try:
            self.log_message("Начинаем обработку видео...")
            self.progress_var.set(5)

            # Устанавливаем проверку тайм-аута *перед* началом каких-либо задач.
            def check_timeout():
                self.check_process_timeout()
                self.root.after(1000, check_timeout)  # Проверяем каждую секунду

            check_timeout()  # Запускаем проверку тайм-аута

            self.progress_var.set(10)
            audio_chunks, total_duration = self.extract_audio_in_chunks(self.source_file.get())
            self.progress_var.set(25)

            segments, total_words_original = self.transcribe_audio_in_chunks(audio_chunks, self.target_language)
            if not segments:
                raise Exception("Транскрипция не удалась.")
            self.progress_var.set(45)

            translated_segments, total_words_translated = self.translate_and_refine_segments(segments)

            # --- Сокращение текста Gemini (необязательно) ---
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
                    self.root.update()  # Обновляем GUI
                translated_segments = shortened_segments

                shortening_change = (
                    ((total_words_translated - total_words_shortened) / total_words_translated) * 100
                    if total_words_translated else 0
                )
                self.log_message(f"Сокращение уменьшило количество слов на: {shortening_change:.2f}%")
            else:
                total_words_shortened = total_words_translated
                self.log_message("Ключ API Gemini не предоставлен. Пропускаем сокращение.")

            self.progress_var.set(60)

            # Создание SRT-файла (пример)
            srt_output_path = os.path.splitext(self.target_file.get())[0] + ".srt"  # Соответствует имени выходного видео
            self.create_srt_file(translated_segments, srt_output_path)

            # --- Отчет о количестве слов ---
            if total_words_original > 0:
                translation_change = (total_words_translated / total_words_original) * 100
                self.log_message(f"Исходное количество слов: {total_words_original}")
                self.log_message(f"Количество переведенных слов: {total_words_translated}")
                self.log_message(f"Изменение количества переведенных слов: {translation_change:.2f}%")
            else:
                self.log_message("Исходное количество слов равно нулю. Пропускаем процент.")

            # --- TTS и объединение аудио ---
            batched_translated_segments = self.batch_segments(translated_segments)
            all_updated_segments = []
            merged_audio_files = []

            for i, batch in enumerate(batched_translated_segments):
                self.log_message(f"Обработка пакета TTS {i+1}/{len(batched_translated_segments)}")
                merged_audio_path, updated_segments = self.generate_tts_audio_for_segments(batch)
                if merged_audio_path:
                    all_updated_segments.extend(updated_segments)
                    merged_audio_files.append(merged_audio_path)
                else:
                    self.log_message(f"Пакет TTS {i+1} не удался.")
                    # Очищаем *все* ранее созданные файлы TTS в случае сбоя
                    for file_path in merged_audio_files:
                        self.remove_file_with_retry(file_path)
                    raise Exception(f"Не удалось создать аудио TTS для пакета {i + 1}.")

            # --- Финальное объединение аудио (если несколько пакетов) ---
            if len(merged_audio_files) > 1:
                final_merged_audio_path = os.path.join(self.process_temp_dir,
                                                        f"final_merged_tts_audio_{timestamp}.mp3")
                self.merge_audio_chunks(merged_audio_files, final_merged_audio_path)
                self.log_message(f"Финальное объединенное аудио TTS: {final_merged_audio_path}")
            elif merged_audio_files:
                final_merged_audio_path = merged_audio_files[0]
                self.log_message(f"Финальное объединенное аудио TTS (один пакет): {final_merged_audio_path}")
            else:
                raise Exception("Аудио TTS не сгенерировано.")

            # --- Очищаем промежуточные файлы TTS ---
            for file_path in merged_audio_files:
                if file_path != final_merged_audio_path:  # Не удаляем финальный файл!
                    self.remove_file_with_retry(file_path)

            self.progress_var.set(80)

            # --- Финальное объединение видео/аудио ---
            self.merge_video_audio(final_merged_audio_path)
            self.progress_var.set(95)

            self.log_message("Обработка успешно завершена! 🎉")
            messagebox.showinfo("Success", "Обработка видео завершена!")


        except TimeoutError as e:
            self.log_message(f"Превышено время ожидания обработки: {str(e)}")
            messagebox.showerror("Error", f"Превышено время ожидания обработки: {str(e)}")
        except Exception as e:
            self.log_message(f"Ошибка: {str(e)}")
            messagebox.showerror("Error", f"Обработка не удалась: {str(e)}")
        finally:
            # --- Очистка ---
            self.progress_var.set(0)
            self.process_start_time = None
            self.current_process = None
            self.start_button.config(state=tk.NORMAL)  # Включаем кнопку
            self.remove_directory_with_retry(self.process_temp_dir)

    def remove_file_with_retry(self, file_path, retries=3, delay=0.5):
        """Удаляет файл, повторяя попытки при необходимости."""
        file_path = os.path.abspath(file_path)  # Используем абсолютный путь
        for i in range(retries):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                self.log_message(f"Файл удален: {file_path}")
                return  # Успех
            except OSError as e:
                if e.errno == errno.ENOENT:  # Файл не найден - уже удален
                    self.log_message(f"Файл не найден (уже удален): {file_path}")
                    return
                if i < retries - 1:  # Не ждем при последней попытке
                    self.log_message(f"Повторная попытка удаления файла ({i+1}/{retries}): {file_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Ошибка при удалении файла после нескольких попыток: {file_path} - {e}")
            except Exception as e:
                self.log_message(f"Непредвиденная ошибка при удалении файла: {file_path} - {e}")
                return  # Не повторяем попытки для непредвиденных ошибок

    def remove_directory_with_retry(self, dir_path, retries=5, delay=1):
        """Удаляет каталог, повторяя попытки при необходимости (особенно для непустых)."""
        dir_path = os.path.abspath(dir_path)
        for i in range(retries):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                self.log_message(f"Каталог удален: {dir_path}")
                return
            except OSError as e:
                if e.errno == errno.ENOENT:  # Каталог уже удален
                    self.log_message(f"Каталог уже удален: {dir_path}")
                    return
                elif e.errno == errno.ENOTEMPTY:  # Каталог не пуст
                    self.log_message(f"Каталог не пуст, повторная попытка удаления ({i+1}/{retries}): {dir_path}")
                    time.sleep(delay)
                else:
                    self.log_message(f"Ошибка при удалении каталога: {dir_path} - {e}")
                    time.sleep(delay)  # Ждем даже для других ошибок
            except Exception as e:
                self.log_message(f"Непредвиденная ошибка при удалении каталога: {dir_path} - {e}")
                return

    def setup_ffmpeg(self):
        """Проверяет доступность FFmpeg."""
        try:
            self.run_subprocess_with_timeout(['ffmpeg', '-version'], 10, "Проверка FFmpeg не удалась")
            self.ffmpeg_available = True
            self.log_message("FFmpeg обнаружен.")
        except FileNotFoundError:
            self.ffmpeg_available = False
            self.log_message("FFmpeg не найден. Установите FFmpeg.")
            messagebox.showwarning("FFmpeg Missing", "Требуется FFmpeg. Пожалуйста, установите его.")
        except Exception as e:
            self.ffmpeg_available = False
            self.log_message(f"Проверка FFmpeg не удалась: {e}")
            messagebox.showwarning("FFmpeg Error", "Проверка FFmpeg не удалась. Проверьте установку.")

    def setup_gui(self):
        """Настраивает графический пользовательский интерфейс."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Заголовок ---
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        title_label = ttk.Label(header_frame, text="🎬 LinguoAI VideoVoice Pro", font=('Helvetica', 16, 'bold'))
        title_label.pack()

        # --- Выбор файлов ---
        file_frame = ttk.LabelFrame(main_frame, text="Видеофайлы", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        # Используем сетку для ввода/вывода
        source_frame = ttk.Frame(file_frame)
        source_frame.pack(fill=tk.X, pady=2)
        ttk.Label(source_frame, text="📹 Вход:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_file, width=40)
        self.source_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(source_frame, text="Обзор", command=self.browse_source, style="Accent.TButton").grid(row=0, column=2, padx=5)

        target_frame = ttk.Frame(file_frame)
        target_frame.pack(fill=tk.X, pady=2)
        ttk.Label(target_frame, text="💾 Выход:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.target_entry = ttk.Entry(target_frame, textvariable=self.target_file, width=40)
        self.target_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(target_frame, text="Обзор", command=self.browse_target, style="Accent.TButton").grid(row=0, column=2, padx=5)

        # Растягиваем поля ввода
        source_frame.columnconfigure(1, weight=1)
        target_frame.columnconfigure(1, weight=1)


        # --- Выбор языка ---
        lang_frame = ttk.LabelFrame(main_frame, text="Настройки голоса", padding="10")
        lang_frame.pack(fill=tk.X, pady=5)

        lang_combo_frame = ttk.Frame(lang_frame)
        lang_combo_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lang_combo_frame, text="🗣️ Целевой язык:").pack(side=tk.LEFT, padx=5)

        # Комбобокс с поиском
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
        self.lang_combo.set("🇬🇧 Английский")  # Устанавливаем значение по умолчанию после создания
        self.language_search_entry.bind("<Return>", (lambda event: self.lang_combo.focus()))
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_language_selected)

        # --- Настройки объединения ---
        options_frame = ttk.LabelFrame(main_frame, text="Настройки объединения", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        # Ключ API Gemini
        brain_frame = ttk.Frame(options_frame)
        brain_frame.pack(fill=tk.X, pady=2)

        # --- Описание API Key Gemini ---
        gemini_description_label = ttk.Label(
            brain_frame,
            wraplength=600,  # Переносим текст
            justify=tk.LEFT,  # Выравниваем по левому краю
            text="Эта программа использует Google Gemini Pro API для необязательного сокращения текста.  "
                 "Это может помочь уменьшить общую длину переведенного текста, сохраняя ключевую информацию.\n"
                 "Для использования этой функции требуется ключ API.  Если у вас нет ключа, вы можете пропустить этот шаг, "
                 "и программа продолжит работу без сокращения."
        )
        gemini_description_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        gemini_link_label = ttk.Label(
            brain_frame,
            text="Получить ключ API Gemini можно здесь: ",
            foreground="dodgerblue",  # Более мягкий синий цвет
            cursor="hand2"  # Меняем курсор при наведении
        )
        gemini_link_label.pack(side=tk.TOP, fill=tk.X)
        gemini_link_label.bind("<Button-1>", lambda e: self.open_webpage("https://makersuite.google.com/app/apikey"))

        self.gemini_key_label = ttk.Label(brain_frame, text="Ключ API Gemini:")
        self.gemini_key_label.pack(side=tk.LEFT, padx=5)
        self.gemini_key_entry = ttk.Entry(brain_frame, textvariable=self.gemini_key, width=30, show="*")
        self.gemini_key_entry.pack(side=tk.LEFT)

        self.start_button = ttk.Button(options_frame, text="🚀 Начать обработку", command=self.start_processing,
                                       style="Success.TButton")
        self.start_button.pack(pady=10)

        # --- Прогресс ---
        progress_frame = ttk.LabelFrame(main_frame, text="Прогресс", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100,
                                            style="Success.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(progress_frame, text="Готов")
        self.status_label.pack()

        # --- Лог ---
        log_frame = ttk.LabelFrame(main_frame, text="Лог обработки", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, state='disabled', wrap=tk.WORD)  # Включаем перенос по словам
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def on_language_selected(self, event):
        """Обрабатывает выбор языка из комбобокса."""
        selected_lang_str = self.lang_combo.get()
        for code, (name, flag) in self.languages.items():
            if f"{flag} {name}" == selected_lang_str:
                self.target_language = code  # Обновляем целевой язык
                self.log_message(f"Выбран целевой язык: {name} ({code})")
                break

    def update_language_list(self, *args):
        """Фильтрует список языков на основе поискового запроса."""
        search_term = self.language_search_var.get().lower()
        filtered_languages = [
            f"{flag} {name}"
            for code, (name, flag) in self.languages.items()
            if search_term in name.lower() or search_term in code.lower() or search_term in flag.lower()
        ]
        self.lang_combo['values'] = filtered_languages
        if filtered_languages:
            self.lang_combo.set(filtered_languages[0])  # Устанавливаем первое совпадение
        else:
            self.lang_combo.set('')  # Очищаем, если нет совпадений
    def browse_source(self):
        """Открывает диалоговое окно выбора файла для выбора исходного видео."""
        filename = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Видеофайлы", "*.mp4 *.avi *.mkv *.mov"), ("Все файлы", "*.*")]
        )
        if filename:
            self.source_file.set(filename)
            base = os.path.splitext(filename)[0]
            self.target_file.set(f"{base}_translated.mp4")
            self.source_entry.xview_moveto(1) #Прокручиваем до конца
            self.target_entry.xview_moveto(1)

    def browse_target(self):
        """Открывает диалоговое окно выбора файла для выбора пути сохранения переведенного видео."""
        filename = filedialog.asksaveasfilename(
            title="Сохранить переведенное видео",
            defaultextension=".mp4",
            filetypes=[("Файлы MP4", "*.mp4")]
        )
        if filename:
            self.target_file.set(filename)
            self.source_entry.xview_moveto(1)  # Прокручиваем до конца
            self.target_entry.xview_moveto(1)
    def log_message(self, message):
        """Записывает сообщение в GUI и внутренний список логов."""
        self.log_messages.append(message)
        self.log_text.config(state='normal')  # Временно включаем
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')  # Снова отключаем
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def load_api_keys_from_environment(self):
        """Загружает ключи API из переменных окружения."""
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_key.set(self.gemini_api_key)  # Загружаем в GUI

    def start_processing(self):
        """Запускает обработку видео в отдельном потоке."""
        if not self.ffmpeg_available:
            messagebox.showerror("Ошибка", "Требуется FFmpeg!")
            return
        if not self.source_file.get() or not self.target_file.get():
            messagebox.showerror("Ошибка", "Выберите входной и выходной файлы.")
            return

        try:
            lang_str = self.lang_combo.get()
            self.target_language = next(
                (code for code, (name, flag) in self.languages.items() if f"{flag} {name}" == lang_str), None
            )
            if self.target_language is None:
                raise ValueError("Выбран неверный язык.")
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.gemini_api_key = self.gemini_key.get()
        if self.gemini_api_key:
            self.init_gemini()
        else:
            self.log_message("Ключ API Gemini не предоставлен. Пропускаем сокращение.")

        self.start_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.log_text.config(state='normal')
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state='disabled')
        self.log_messages = []
        threading.Thread(target=self.process_video, daemon=True).start()

    def load_whisper_model(self):
        """Загружает модель Faster Whisper."""
        try:
            self.log_message(f"Загрузка модели Whisper ({self.model_size}) на {self.device}...")
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.log_message("Модель Whisper успешно загружена.")
        except Exception as e:
            self.log_message(f"Ошибка при загрузке модели Whisper: {e}")
            raise

    def extract_audio_in_chunks(self, video_path: str) -> Tuple[List[str], float]:
        """Извлекает аудио из видео фрагментами."""
        self.log_message(f"Извлечение аудио из: {video_path}")
        audio_chunks = []
        try:
            duration_command = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            duration_output, _ = self.run_subprocess_with_timeout(duration_command, 30, "Не удалось получить длительность видео")
            total_duration = float(duration_output.strip())
            self.log_message(f"Общая длительность видео: {total_duration:.2f} секунд")

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
            self.log_message(f"Ошибка во время извлечения аудио: {e}")
            raise  # Повторно вызываем исключение для обработки в process_video

    def transcribe_audio_in_chunks(self, audio_chunks: List[str], language: str) -> Tuple[List[Dict], int]:
        """Транскрибирует аудиофрагменты с использованием Faster Whisper."""
        all_segments = []
        total_words = 0
        if self.whisper_model is None:
            raise ValueError("Модель Whisper не загружена.")

        try:
            for i, chunk_path in enumerate(audio_chunks):
                self.log_message(f"Транскрибирование фрагмента {i+1}/{len(audio_chunks)}: {chunk_path}")
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
                self.log_message(f"Транскрипция фрагмента {i+1} завершена.")
                self.root.update()  # Обновляем GUI

                self.remove_file_with_retry(chunk_path)
                self.remove_directory_with_retry(os.path.dirname(chunk_path))

            self.log_message(f"Общее количество транскрибированных слов: {total_words}")
            return all_segments, total_words
        except Exception as e:
            self.log_message(f"Ошибка во время транскрипции: {e}")
            raise

    def translate_and_refine_segments(self, segments: List[Dict]) -> Tuple[List[Dict], int]:
        """Переводит сегменты и обрабатывает возможные ошибки перевода."""
        translated_segments = []
        total_words_translated = 0
        translator = GoogleTranslator(source='auto', target=self.target_language)

        try:
            for segment in segments:
                self.log_message(f"Перевод сегмента: '{segment['text'][:50]}...' на {self.target_language}")
                translated_text = translator.translate(segment['text'])
                if not isinstance(translated_text, str):
                    self.log_message(
                        f"ПРЕДУПРЕЖДЕНИЕ: translator.translate не вернул строку. Тип: {type(translated_text)}, Значение: {translated_text}")
                    translated_text = ""  # Устанавливаем пустую строку
                translated_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': translated_text
                })
                total_words_translated += len(translated_text.split())
                self.log_message(f"Переведенный сегмент: '{translated_text[:50]}...'")
            return translated_segments, total_words_translated
        except Exception as e:
            self.log_message(f"Ошибка во время перевода: {e}")
            raise

    def generate_tts_audio_for_segments(self, translated_segments: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
        """Создает аудио TTS для пакета переведенных сегментов."""
        tts_chunks = []
        updated_segments = []
        total_duration = 0
        try:
            for i, segment in enumerate(translated_segments):
                text = segment['text']
                self.log_message(f"Создание TTS для сегмента {i+1}/{len(translated_segments)}: '{text[:50]}...'")
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
                                                                        "Не удалось получить длительность сегмента TTS")
                    segment_duration = float(duration_output.strip())
                    updated_segments.append({
                        'start': total_duration,
                        'end': total_duration + segment_duration,
                        'text': text,
                    })
                    total_duration += segment_duration
                except Exception as e:
                    self.log_message(f"Ошибка при создании TTS для сегмента {i + 1}: {e}")
                    # Очищаем все созданные файлы в этом пакете на данный момент
                    for file in tts_chunks:
                        self.remove_file_with_retry(file)
                        self.remove_directory_with_retry(os.path.dirname(file))  # Пытаемся удалить каталог
                    return None, []  # Указываем на неудачу

                self.log_message(f"TTS для сегмента {i+1} создан.")
                self.root.update()  # Поддерживаем отзывчивость GUI

            self.log_message(f"Объединение {len(tts_chunks)} фрагментов TTS...")
            if tts_chunks:
                merged_audio_path = os.path.join(self.process_temp_dir, f"merged_tts_audio_{int(time.time())}.mp3")
                self.merge_audio_chunks(tts_chunks, merged_audio_path)
                self.log_message(f"Объединенное аудио TTS: {merged_audio_path}")

                for tts_file in tts_chunks:
                    self.remove_file_with_retry(tts_file)
                    self.remove_directory_with_retry(os.path.dirname(tts_file))

                return merged_audio_path, updated_segments
            else:
                self.log_message("Сегменты TTS не созданы.")
                return None, []

        except Exception as e:
            self.log_message(f"Ошибка во время создания/объединения TTS: {e}")
            for file in tts_chunks:  # Исправлено
                self.remove_file_with_retry(file)
                self.remove_directory_with_retry(os.path.dirname(file))  # и каталог
            return None, []

    def open_webpage(self, url):
        """Открывает веб-страницу в браузере по умолчанию."""
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
# --- КОНЕЦ ФАЙЛА video.py ---