#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sayan112207/Perplexa/blob/main/Perplexa_Research_Mode.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


get_ipython().system('pip install arxiv==2.1.3 llama_index==0.12.3 llama-index-llms-mistralai==0.3.0 llama-index-embeddings-mistralai==0.3.0 gradio==3.39.0')


# In[2]:


from getpass import getpass
import requests
import sys
import arxiv
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, load_index_from_storage, PromptTemplate, Settings
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent


# In[3]:


from google.colab import userdata
api_key = userdata.get('MISTRAL_API_KEY')


# In[4]:


llm = MistralAI(api_key=api_key, model='mistral-large-latest')


# In[5]:


model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=api_key)


# In[6]:


def fetch_arxiv_papers(title :str, papers_count: int):
    search_query = f'all:"{title}"'
    search = arxiv.Search(
        query=search_query,
        max_results=papers_count,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    papers = []
    # Use the Client for searching
    client = arxiv.Client()

    # Execute the search
    search = client.results(search)

    for result in search:
        paper_info = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'published': result.published,
                'journal_ref': result.journal_ref,
                'doi': result.doi,
                'primary_category': result.primary_category,
                'categories': result.categories,
                'pdf_url': result.pdf_url,
                'arxiv_url': result.entry_id
            }
        papers.append(paper_info)

    return papers

papers = fetch_arxiv_papers("Language Models", 10)


# In[7]:


[[p['title']] for p in papers]


# In[8]:


def create_documents_from_papers(papers):
    documents = []
    for paper in papers:
        content = f"Title: {paper['title']}\n"                   f"Authors: {', '.join(paper['authors'])}\n"                   f"Summary: {paper['summary']}\n"                   f"Published: {paper['published']}\n"                   f"Journal Reference: {paper['journal_ref']}\n"                   f"DOI: {paper['doi']}\n"                   f"Primary Category: {paper['primary_category']}\n"                   f"Categories: {', '.join(paper['categories'])}\n"                   f"PDF URL: {paper['pdf_url']}\n"                   f"arXiv URL: {paper['arxiv_url']}\n"
        documents.append(Document(text=content))
    return documents



#Create documents for LlamaIndex
documents = create_documents_from_papers(papers)


# In[9]:


Settings.chunk_size = 1024
Settings.chunk_overlap = 50

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)


# In[10]:


index.storage_context.persist('index/')
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir='index/')

#load index
index = load_index_from_storage(storage_context, embed_model=embed_model)


# In[11]:


query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

rag_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="research_paper_query_engine_tool",
    description="A RAG engine with recent research papers.",
)


# In[12]:


from llama_index.core import PromptTemplate
from IPython.display import Markdown, display
# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}" f"**Text:** "
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown(""))

prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)


# In[13]:


def download_pdf(pdf_url, output_file):
    """
    Downloads a PDF file from the given URL and saves it to the specified file.

    Args:
        pdf_url (str): The URL of the PDF file to download.
        output_file (str): The path and name of the file to save the PDF to.

    Returns:
        str: A message indicating success or the nature of an error.
    """
    try:
        # Send a GET request to the PDF URL
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Write the content of the PDF to the output file
        with open(output_file, "wb") as file:
            file.write(response.content)

        return f"PDF downloaded successfully and saved as '{output_file}'."

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"


# In[14]:


download_pdf_tool = FunctionTool.from_defaults(
    download_pdf,
    name='download_pdf_file_tool',
    description='python function, which downloads a pdf file by link'
)
fetch_arxiv_tool = FunctionTool.from_defaults(
    fetch_arxiv_papers,
    name='fetch_from_arxiv',
    description='download the {max_results} recent papers regarding the topic {title} from arxiv'
)


# In[15]:


# building an ReAct Agent with the three tools.
agent = ReActAgent.from_tools([download_pdf_tool, rag_tool, fetch_arxiv_tool], llm=llm, verbose=True)


# In[16]:


# create a prompt template to chat with an agent
q_template = (
    "I am interested in {topic}. \n"
    "Find papers in your knowledge database related to this topic; use the following template to query research_paper_query_engine_tool tool: 'Provide title, summary, authors and link to download for papers related to {topic}'. If there are not, could you fetch the recent one from arXiv? \n"
)


# In[17]:


answer = agent.chat(q_template.format(topic="Audio-Language Models"))


# In[18]:


Markdown(answer.response)


# In[19]:


answer = agent.chat("Download the papers, which you mentioned above")


# In[20]:


Markdown(answer.response)


# In[21]:


answer = agent.chat(q_template.format(topic="Min Max Similarity"))


# In[22]:


Markdown(answer.response)


# In[25]:


import gradio as gr

def research_agent(topic):
    """
    Function to handle user queries, interact with the agent,
    and return the agent's response.

    Args:
        topic (str): The user's research topic.

    Returns:
        str: The agent's response.
    """

    answer = agent.chat(q_template.format(topic=topic))  # Get the agent's response
    return (answer.response)  # Return the response


# In[26]:


iface = gr.Interface(
    fn=research_agent,  # The function to handle queries
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your research topic here..."),  # Input field
    outputs="text",  # Output format
    title="Research Paper Agent",  # Title of the interface
    description="Explore recent research papers on various topics.",  # Description
)


# In[27]:


iface.launch(share=True, debug=True)


# In[ ]:




