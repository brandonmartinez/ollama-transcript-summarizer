import os
import mdformat
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import TokenTextSplitter
from langchain.globals import set_debug
import logging
import sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# set_debug(True)


class TranscriptSummarizer():
    def __init__(self, model="llama3"):
        host = os.getenv('OLLAMA_HOST', "localhost:11434")
        base_url = f"http://{host}"
        self.model = Ollama(base_url=base_url, model=model)
        self.output_parser = StrOutputParser()

    def summarize(self, text: str):
        # Split text into chunks
        ##################################################
        logging.info('Splitting text into chunks')
        text_splitter = TokenTextSplitter(
            # Controls the size of each chunk
            chunk_size=1000,
            # Controls overlap between chunks
            chunk_overlap=20,
        )

        text_documents = text_splitter.create_documents([text])
        logging.info('Documents chunked')
        logging.debug(text_documents)

        # Combine Documents
        ##################################################
        text_document_summary_prompt = PromptTemplate.from_template(
            """You are an expert at summarizing conversation transcripts.
Given the following conversation transcript, capture the most important topics and details
from the conversation. Your summary should be concise and to the point, and be written
in a way that can be easily understood by someone who was not present during the conversation.
The summary should be at least three sentences long. Don't include anything that is
not present in the conversation transcript.

TRANSCRIPT:
{page_content}

SUMMARY:
""")

        text_document_summary_chain = (
            text_document_summary_prompt | self.model | self.output_parser
        )

        logging.info('Executing text_document_summary_chain')
        text_document_summary_output = text_document_summary_chain.batch(
            text_documents)

        logging.info('Summaries returned')
        logging.debug(text_document_summary_output)

        # Create summary
        ##################################################
        summary_prompt = PromptTemplate.from_template(
            """You are an expert at taking summaries of a conversation transcript, combining them, and retrieving the most important information.
Only use information that is provided by the transcript summaries.
The summary should capture key points and important details from the conversation.
The summary should be concise and to the point, but should be at least ten sentences long.
Do not make up any information; only return data that is present in the provided text.
Here is a short example of the format of the summary, your response should be longer than this:

EXAMPLE OUTPUT:
=====

Summary:
Here is a detailed summary of the conversation. It should have enough information to give a good overview of the conversation.
It will include multiple sentences to cover all the important points from the conversation.

Important Points:
- **Point:** This is important
- **Another point:** This is also important
- **Yet another point:** This is equally as important

=====
CONTEXT:
{context}

SUMMARY:
""")

        summary_chain = (
            summary_prompt | self.model | self.output_parser
        )

        logging.info('Executing summary_chain')
        output = summary_chain.invoke(
            {"context": text_document_summary_output})
        formatted_text = mdformat.text(output)
        logging.info('Final summary returned')
        logging.debug(formatted_text)

        return formatted_text
