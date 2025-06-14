from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
import fitz  # PyMuPDF
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
import os

# Set API keys for OpenAI and SerperDev
os.environ[
    "OPENAI_API_KEY"] = "eueren Key einfügen"
os.environ["SERPERDEV_API_KEY"] = "euren Key einfügen"

# Create Documents for information retrieval.
document_store = InMemoryDocumentStore()

# PDF-Verzeichnis
pdf_folder = "data/Datenbank"

documents = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        doc_path = os.path.join(pdf_folder, filename)
        pdf = fitz.open(doc_path)
        text = ""
        for page in pdf:
            text += page.get_text()
        documents.append(Document(content=text, meta={"source": filename}))
        pdf.close()

document_store = InMemoryDocumentStore()
document_store.write_documents(documents)

# Initialize the retriever component.
retriever = InMemoryBM25Retriever(document_store=document_store)

# Define a prompt template for the retrieval-based answer generation
prompt_template = [
    ChatMessage.from_user(
        """
You are only allowed to answer using the provided documents.
If the answer is not found there, simply respond with: no_answer

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}

Query: {{query}}
"""
    )
]

# Build the prompt using the above template.
prompt_builder = ChatPromptBuilder(template=prompt_template, required_variables="*")
# Initialize the LLM for generating answers from documents.
llm = OpenAIChatGenerator(
    model="meta-llama/Llama-3-70b-chat-hf",
    api_base_url="https://api.together.xyz/v1",
    api_key=Secret.from_token("f685fe1a0cd83ee4e3d018ef9f5f21b7da924b3a62c4d7b4a57c465554d267c1")
)

# Define a prompt template for answers generated from web search.
web_prompt_template = [
    ChatMessage.from_user("""
Answer the following query given the documents retrieved from the web.
Your answer should indicate that your answer was generated from websearch.

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}
Query: {{query}}
""")
]

# Build the web prompt.
prompt_builder_web = ChatPromptBuilder(template=web_prompt_template, required_variables="*")
# Initialize the LLM for web-based answers.
llm_web = OpenAIChatGenerator(
    model="meta-llama/Llama-3-70b-chat-hf",
    api_base_url="https://api.together.xyz/v1",
    api_key=Secret.from_token("f685fe1a0cd83ee4e3d018ef9f5f21b7da924b3a62c4d7b4a57c465554d267c1")
)
websearch = SerperDevWebSearch()  # Web search component using SerperDev.

# Define the routing logic.
# If the local model says "no_answer", route the query to web search.
routes = [
    {
        "condition": "{{'no_answer' in replies[0].text}}",
        "output": "{{query}}",
        "output_name": "go_to_websearch",
        "output_type": str,
    },
    {
        "condition": "{{'no_answer' not in replies[0].text}}",
        "output": "{{replies[0].text}}",
        "output_name": "answer",
        "output_type": str,
    },
]
router = ConditionalRouter(routes)

# Build the overall pipeline and connect components.
pipeline = Pipeline()
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm)
pipeline.add_component("router", router)
pipeline.add_component("websearch", websearch)
pipeline.add_component("prompt_builder_web", prompt_builder_web)
pipeline.add_component("llm_web", llm_web)

# Define the data flow between pipeline components.
pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.messages")
pipeline.connect("llm.replies", "router.replies")
pipeline.connect("router.go_to_websearch", "websearch.query")
pipeline.connect("router.go_to_websearch", "prompt_builder_web.query")
pipeline.connect("websearch.documents", "prompt_builder_web.documents")
pipeline.connect("prompt_builder_web", "llm_web")


def main():
    query = input("Deine Frage: ")

    # Run the pipeline with the user query.
    result = pipeline.run({
        "retriever": {"query": query},
        "prompt_builder": {"query": query},
        "router": {"query": query}
    })
    # Print either the local answer or the web-based answer.
    if "router" in result and "answer" in result["router"]:
        print(result["router"]["answer"])
    else:
        print("Web-Antwort:")
        print(result["llm_web"]["replies"][0].text)


if __name__ == "__main__":
    main()
