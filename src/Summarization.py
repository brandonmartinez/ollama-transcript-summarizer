import os
import mdformat
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        text_splitter = RecursiveCharacterTextSplitter(
            # Controls the size of each chunk
            chunk_size=1000,
            # Controls overlap between chunks
            chunk_overlap=200,
        )

        text_documents = text_splitter.create_documents([text])
        text_documents = [doc for doc in text_documents if not doc.page_content.endswith(":")]
        logging.info('Documents chunked')
        logging.debug(text_documents)

        # Combine Documents
        ##################################################
        text_document_summary_prompt = PromptTemplate.from_template(
            """You are an expert at summarizing portions of a conversation transcript.
Given the following section of a transcript, rewrite it to be more concise and to the the point.
Do not make up any information; only use information that is provided by the transcript.
If there is nothing to summarize, output an empty string. Do not include the prompt in your response.
Do not state what you are responding with, just the response itself. Do not address the conversation, just state the information.

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
Do not make up any information; only use information that is provided by the transcript summaries.
The summary should capture key points and important details from the conversation.
The summary should be concise and to the point, but should be long enough to form a blog post.
Here is a short example of the format of the summary, your response should be longer than this:

TRANSCRIPT SUMMARIES:
{context}
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
