#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SOURCE_DIR="$1"
CUSTOM_PROMPT="${2:-}"
CURRENT_WORKING_DIR=$(dirname "$0")
TEMP_DIR="${SOURCE_DIR}/_temp"
OUTPUT_DIR="${SOURCE_DIR}/_output"
TEMPTEXT_FILE="$TEMP_DIR/temp.txt"
TRANSCRIPT_FILE="${OUTPUT_DIR}/transcript.txt"

mkdir -p "${TEMP_DIR}" "${OUTPUT_DIR}"

if ! command -v whisper &> /dev/null
then
    echo "whisper could not be found"
    echo "Please install it by following the instructions at https://github.com/openai/whisper/#setup"
    exit
fi

if ! command -v ollama &> /dev/null
then
    echo "ollama could not be found"
    echo "Please install it by following the instructions at https://github.com/ollama/ollama"
    exit
fi

echo "Processing audio files..."

if [ -z "${SKIP_TRANSCRIPTION+x}" ] || [ "${SKIP_TRANSCRIPTION}" = "0" ] || [ "${SKIP_TRANSCRIPTION}" = "false" ]; then

    # Find all audio files in the source directory
    while IFS=  read -r -d $'\0'; do
        AUDIO_FILES+=("$REPLY")
    done < <(find "$SOURCE_DIR" -type f \( -iname \*.wav -o -iname \*.mp3 -o -iname \*.aiff -o -iname \*.m4a -o -iname \*.flac \) -print0)
    echo "Found ${#AUDIO_FILES[@]} audio files"

    # check to see if there are multiple files with the same base name
    # if so, only keep the wav file (this skips conversion of the same original file)
    AUDIO_FILE_MAP=()
    AUDIO_FILE_MAP_KEYS=()

    for AUDIO_FILE in "${AUDIO_FILES[@]}"
    do
        AUDIO_FILE_BASENAME=$(basename "$AUDIO_FILE")
        AUDIO_FILE_BASENAME_NO_EXT="${AUDIO_FILE_BASENAME%.*}"

        echo "Validating if $AUDIO_FILE_BASENAME is unique"

        KEY_EXISTS=0
        if [ ${#AUDIO_FILE_MAP_KEYS[@]} -ne 0 ]; then
            for KEY in ${AUDIO_FILE_MAP_KEYS[@]}; do
                if [[ "$KEY" == "$AUDIO_FILE_BASENAME_NO_EXT" ]]; then
                    KEY_EXISTS=1
                    break
                fi
            done
        fi

        if [[ "$KEY_EXISTS" -eq 1 ]]; then
            if [[ "${AUDIO_FILE##*.}" == "wav" ]]; then
                for i in "${!AUDIO_FILE_MAP[@]}"; do
                    if [[ "${AUDIO_FILE_MAP_KEYS[$i]}" == "$AUDIO_FILE_BASENAME_NO_EXT" ]]; then
                        AUDIO_FILE_MAP[$i]="$AUDIO_FILE"
                    fi
                done
            fi
        else
            AUDIO_FILE_MAP+=("$AUDIO_FILE")
            AUDIO_FILE_MAP_KEYS+=("$AUDIO_FILE_BASENAME_NO_EXT")
        fi
    done

    AUDIO_FILES=("${AUDIO_FILE_MAP[@]}")

    # Process each file
    for AUDIO_FILE in "${AUDIO_FILES[@]}"
    do
        echo "Processing $AUDIO_FILE"

        FILE_EXTENSION="${AUDIO_FILE##*.}"
        if [ "$FILE_EXTENSION" != "wav" ]; then
            echo "Converting $AUDIO_FILE to wave format"
            CONVERTED_AUDIO_FILE="${AUDIO_FILE%.*}.wav"
            ffmpeg -y -i "$AUDIO_FILE" "$CONVERTED_AUDIO_FILE"
            AUDIO_FILE="$CONVERTED_AUDIO_FILE"
        fi

        echo "Transcribing $AUDIO_FILE with OpenAI whisper"
        AUDIO_FILE_BASENAME=$(basename "$AUDIO_FILE" .wav)

        AUDIO_FILE_NAME=$(echo "$AUDIO_FILE_BASENAME" | awk -F- '{print $NF}')
        AUDIO_FILE_TRANSCRIPT_NAME="$TEMP_DIR/$AUDIO_FILE_NAME.txt"
        whisper "$AUDIO_FILE" --model "tiny.en" --output_dir "$TEMP_DIR" --output_format vtt --language en --device cpu --fp16 False > "$AUDIO_FILE_TRANSCRIPT_NAME"

        # Append audio file name name to timestamps using awk
        awk -v audiofilename="$AUDIO_FILE_NAME" '{print $0 " [ " audiofilename " ]"}' "$AUDIO_FILE_TRANSCRIPT_NAME" > "$TEMPTEXT_FILE" && mv "$TEMPTEXT_FILE" "$AUDIO_FILE_TRANSCRIPT_NAME"
    done

    echo "Combining transcripts into a single file"
    cat "${TEMP_DIR}"/*.txt | sort > "$TRANSCRIPT_FILE"

    echo "Removing timestamps from transcript"

    sed -r -i'.bak' -e 's/\[[0-9]{2}\:[0-9]{2}\.[0-9]{3}\ \-\-\>\ [0-9]{2}\:[0-9]{2}\.[0-9]{3}\]\ \ //' "$TRANSCRIPT_FILE"

    echo "Moving audio file name to the beginning of the line"

    sed -r -i'.bak' -e 's/(.*)(\[ ?[A-Za-z ]+ ?\])$/\2 \1/' "$TRANSCRIPT_FILE"

    rm "$TRANSCRIPT_FILE.bak"
fi

LLM_PROMPT="Rewrite the following into a formal summary of the conversation, making sure to include important details and explanations of topics discussed. Remove any names and rewrite into a third person perspective that is speaking about the solution. $CUSTOM_PROMPT"
LLM_PROMPT="$LLM_PROMPT\n\n$(cat "$TRANSCRIPT_FILE")"
LLM_FORMAT_PROMPT="Create a markdown-formatted post. The following text should be a summary of the highlights, action items, and follow ups from the conversation broken into sections and a bulleted list."
# Models pulled from this library: https://ollama.com/library
LLM_MODELS=("llama2" "mistral" "openchat" "llama2-uncensored")

if [ -z "${UPDATE_MODELS+x}" ]; then
    UPDATE_MODELS=0
fi

for LLM_MODEL in "${LLM_MODELS[@]}"
do
    if [ "${UPDATE_MODELS}" = "1" ] || [ "${UPDATE_MODELS}" = "true" ]; then
        echo "Pulling latest $LLM_MODEL model"
        ollama pull $LLM_MODEL
    fi

    SUMMARY_FILE="${OUTPUT_DIR}/summary_${LLM_MODEL}.txt"
    HIGHLIGHTS_FILE="${OUTPUT_DIR}/highlights_${LLM_MODEL}.md"

    echo "Running $LLM_MODEL model"
    ollama run $LLM_MODEL "$LLM_PROMPT" > "$SUMMARY_FILE"

    echo "Summarizing with $LLM_MODEL model"
    LLM_SPECIFIC_FORMAT_PROMPT="$LLM_FORMAT_PROMPT\n\n$(cat "$SUMMARY_FILE")"
    ollama run $LLM_MODEL "$LLM_SPECIFIC_FORMAT_PROMPT" > "$HIGHLIGHTS_FILE"

    echo "Formatting markdown"
    prettier --config "$CURRENT_WORKING_DIR/.prettierrc" --write "$HIGHLIGHTS_FILE"

    echo "Finished processing with $LLM_MODEL model; output written to $SUMMARY_FILE and $HIGHLIGHTS_FILE."

done

echo "Done!"
