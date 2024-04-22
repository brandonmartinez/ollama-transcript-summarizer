import os
import whisper
import subprocess


class Converter:
    def __init__(self, ffmpeg_parameters: str = "-ar 16000 -ac 1 -c:a pcm_s16le"):
        self.ffmpeg_parameters = ffmpeg_parameters

    def convert(self, file_path: str, target_directory: str = None) -> str:
        file_name = file_path.split("/")[-1].split(".")[0]
        if target_directory is None:
            target_directory = "/".join(file_path.split("/")[:-1])
        target_file_path = target_directory + "/" + file_name + ".wav"

        if os.path.exists(target_file_path):
            print("Target file already exists:", target_file_path)
            return target_file_path

        cmd = "ffmpeg -i '" + file_path + "' " + \
            self.ffmpeg_parameters + " '" + target_file_path + "'"

        subprocess.run(cmd, shell=True)

        return target_file_path


class Transcriber:
    def __init__(self, model_name: str = "tiny.en"):
        self.model = whisper.load_model(model_name)

    def process_segments(self, segments: list, transcript_file: str):
        transcript_file_name = transcript_file.split("/")[-1].split(".")[0]

        lines = []
        for segment in segments:
            # Whisper will sometimes return the same line multiple times, skip those
            if lines and segment["text"] == lines[-1].split(" | ")[-1]:
                continue

            line_text = f'{segment["start"]:09.3f} | {segment["end"]:09.3f} | {transcript_file_name} | {segment["text"]}'
            lines.append(line_text)

        with open(transcript_file, 'w') as file:
            file.write('\n'.join(lines))

    def transcribe(self, audio_file: str, transcript_directory: str = None) -> str:
        if transcript_directory is None:
            transcript_directory = "/".join(audio_file.split("/")[:-1])
        audio_file_name = audio_file.split("/")[-1].split(".")[0]
        transcript_file = transcript_directory + "/" + audio_file_name + ".txt"

        if os.path.exists(transcript_file):
            print("Transcript file already exists:", transcript_file)
            return transcript_file

        result = self.model.transcribe(audio=audio_file, verbose=True)
        self.process_segments(result["segments"], transcript_file)
        return transcript_file

    def combine(self, transcript_files: list[str], transcript_directory: str = None) -> str:
        if transcript_directory is None:
            transcript_directory = "/".join(transcript_files[0].split("/")[:-1])
        combined_transcript_file = transcript_directory + "/_combined.txt"

        if os.path.exists(combined_transcript_file):
            print("Combined transcript file already exists:", combined_transcript_file)
            return combined_transcript_file

        combined_transcripts = []
        for transcript_file in transcript_files:
            with open(transcript_file, 'r') as file:
                lines = file.readlines()
                combined_transcripts.extend(lines)

        combined_transcripts.sort()

        modified_transcripts = []
        for line in combined_transcripts:
            modified_line = " | ".join(line.split(" | ")[2:])
            modified_transcripts.append(modified_line)

        with open(combined_transcript_file, 'w') as file:
            file.writelines(modified_transcripts)

        return combined_transcript_file
