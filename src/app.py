import os
import Transcription
import Summarization
import sys

# setup working variables
source_directory = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "_temp"
target_directory = source_directory + "/_output"
converted_directory = target_directory + "/_converted"
transcript_directory = target_directory + "/_transcripts"

# initialize helpers
converter = Transcription.Converter()
transcriber = Transcription.Transcriber()
summarizer = Summarization.TranscriptSummarizer()

# Create directories
os.makedirs(target_directory, exist_ok=True)
os.makedirs(converted_directory, exist_ok=True)
os.makedirs(transcript_directory, exist_ok=True)

# gather files
source_files = [file for file in os.listdir(source_directory) if os.path.isfile(
    os.path.join(source_directory, file)) and file.lower().endswith(('.wav', '.aiff', '.mp3', '.flac', '.m4a'))]

# convert all files
converted_files = list[str]()
for source_file in source_files:
    source_file_path = source_directory + "/" + source_file
    converted_file_path = converter.convert(
        source_file_path, target_directory=converted_directory)
    converted_files.append(converted_file_path)

# transcribe all files
transcribed_files = list[str]()
for converted_file in converted_files:
    transcribed_file = transcriber.transcribe(
        converted_file, transcript_directory=transcript_directory)
    transcribed_files.append(transcribed_file)

# combine all transcriptions
combined_transcripts = transcriber.combine(
    transcribed_files, transcript_directory=transcript_directory)

with open(combined_transcripts, 'r') as file:
    transcript_text = file.read()

summary = summarizer.summarize(transcript_text)

summary_file_path = target_directory + "/_summary.txt"
with open(summary_file_path, 'w') as file:
    file.write(summary)

