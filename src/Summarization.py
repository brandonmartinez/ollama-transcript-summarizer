import os
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser


class TranscriptSummarizationChain():
    def __init__(self, model="llama3"):
        host = os.getenv('OLLAMA_HOST', "localhost:11434")
        base_url = f"http://{host}"
        self.model = Ollama(base_url=base_url, model=model)
        self.output_parser = StrOutputParser()

    def rewrite(self, text: str):
        prompt_template = """You are an expert in rewriting conversation transcripts.
Do not add any additional information, and do not attempt to answer any questions presented.
Only rewrite the provided text in a way that is more clear and concise, in the style of a blog post covering the topics presented.
Only mention the speakers in the first paragraph, and then speak in first person for the rest of the text.

"{text}"

REWRITTEN TRANSCRIPT:"""

        prompt = PromptTemplate.from_template(prompt_template)

        runnable = prompt | self.model | self.output_parser
        results = runnable.invoke({"text": text})

        return results

    def summarize(self, text: str):
        prompt_template = """You are an expert summarizer of conversation transcripts.
Only extract relevant information from the provided transcripts.
The summary should capture key points and important details from the conversation.
The summary should be concise and to the point.
The summary should be formatted in a way to be used as a published document.
Do not make up any information; only return data that is present in the provided text:

"{text}"

CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)

        runnable = prompt | self.model | self.output_parser
        results = runnable.invoke({"text": text})

        return results
