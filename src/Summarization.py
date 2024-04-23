import os
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import TokenTextSplitter
from langchain.globals import set_debug

set_debug(True)


class TranscriptSummarizer():
    def __init__(self, model="llama3"):
        host = os.getenv('OLLAMA_HOST', "localhost:11434")
        base_url = f"http://{host}"
        self.model = Ollama(base_url=base_url, model=model)
        self.output_parser = StrOutputParser()

    def summarize(self, text: str):
        # Split text into chunks
        ##################################################
        text_splitter = TokenTextSplitter(
            # Controls the size of each chunk
            chunk_size=5000,
            # Controls overlap between chunks
            chunk_overlap=20,
        )

        text_documents = text_splitter.create_documents([text])

        # Combine Documents
        ##################################################
        combine_prompt = PromptTemplate.from_template(
            """You are an expert in summarizing conversation transcripts.
Take the following documents and create summaries of each of them:

{context}
""")

        combine_chain = (
            combine_prompt | self.model | self.output_parser
        )

        # Create summary
        ##################################################
        summary_prompt = PromptTemplate.from_template(
            """You are an expert summarizer of conversation transcripts.
Only extract relevant information from the provided transcripts.
The summary should capture key points and important details from the conversation.
The summary should be concise and to the point, but should be at least three sentences long.
Do not make up any information; only return data that is present in the provided text:
Here is a short example of the format of the summary, your response should be longer than this:

Summary:
Here is a detailed summary of the conversation. It should have enough information to give a good overview of the conversation.

Important Points:
- **Point:** This is important
- **Another point:** This is also important

=====
CONTEXT:
{combine}
""")
        summary_chain = (
            summary_prompt | self.model | self.output_parser
        )
        # Combine chains
        ##################################################
        chain = (
            {
                "combine": combine_chain,
            }
            | summary_chain
            | self.output_parser
        )

        output = chain.invoke({"context": text_documents})
        return output
