import streamlit as st
from streamlit_mic_recorder import mic_recorder,speech_to_text

import os
import boto3
from datetime import datetime
import re
from urllib.parse import urlparse
from pinecone import Pinecone, ServerlessSpec
from langchain_voyageai import VoyageAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain, RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

from ragas.metrics import context_relevancy, answer_relevancy, faithfulness, context_recall, answer_correctness
from ragas.metrics.critique import harmfulness
from ragas import evaluate
from datasets import Dataset

from dotenv import load_dotenv
load_dotenv()

red_square = "\U0001F7E5"
microphone = "\U0001F3A4"
play_button = "\U000025B6"


# Set Streamlit page configuration
st.set_page_config(page_title="💊🩹DoseBuddy Voice Assistant🤖", layout="centered")

## Hide header
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("💊🩹DoseBuddy Voice Assistant🤖")
st.markdown(
    """ 
        > :black[**A Chatbot for Drugs and Supplements,** *powered by -  [LangChain](https://python.langchain.com/v0.2/docs/introduction/) + 
        [OpenAI](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4) + 
        [Streamlit](https://streamlit.io) + 
        [MedlinePlus](https://medlineplus.gov/druginformation.html) + 
        💪 **Griffin Qin***]
        """
)

# Set up sidebar with various options

st.sidebar.warning(
        "The information provided by this chatbot is intended for reference purposes only. \
        It should not be considered as medical advice or a substitute for professional consultation, diagnosis, or treatment.\n \
        \nAlways seek the guidance of a qualified healthcare provider with any questions you may have regarding a medical condition or treatment.\n \
        \nIn case of a medical emergency, immediately contact your doctor or call emergency services. This chatbot does not replace the need for professional medical advice, diagnosis, or treatment."
    ,icon="⚠️")

with st.sidebar.expander(" 🛠️ Settings ", expanded=False):


    MODEL = st.selectbox(
        label="LLM_Model",
        options=[
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
    )
    
    K = st.number_input(
        " (#)Round of chats to consider", min_value=3, max_value=1000
    )



# Config

## Embedding
embeddings = VoyageAIEmbeddings(
    batch_size=64,
    model=os.getenv('EMBEDDING_MODEL_NAME'),  
    voyage_api_key=os.getenv('VOYAGE_AI_API_KEY')

)

## PINECONE
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = "brave-project"

## Retriever
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

## Initialize LLM
llm = ChatOpenAI(model=MODEL)
llm_decide = ChatOpenAI(model="gpt-4-turbo")
llm_gt = ChatOpenAI(model="gpt-4o")

# Function to generate pre-signed URL
def generate_presigned_url(s3_uri):
    # Parse the S3 URI
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip('/')

    # Configure S3
    s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('aws_access_key_id'),
    aws_secret_access_key=os.getenv('aws_secret_access_key'),
    region_name=os.getenv('region_name')
    )
    
    # Generate a pre-signed URL for the S3 object
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': object_key},
        ExpiresIn=3600  # URL expiration time in seconds
    )
    return presigned_url


def get_answer(user_query, chat_history):
    global tool
    global context
    global formulated_question

    tool = ''
    context = ''
    formulated_question = ''

    def inspect_state(state):
        """Print the state passed between Runnables in a langchain and pass it on"""

        print(f"Question from Decide Chain is: {state['question']}")
        return state

    def inspect_topic(state):
        """Print the state passed between Runnables in a langchain and pass it on"""
        global tool
        tool = state['topic']
        print(f"Question from Full Chain is: {state['question']}")
        return state

    def inspect_retrieved_context(state):
        """Print the state passed between Runnables in a langchain and pass it on"""
        global context
        global formulated_question

        context = state['retrieved_context']
        formulated_question = state['question']
        print(f"Question from RAG Chain is: {state['question']}")
        print('RAG Chain Triggered')
        return state
    
    # Create contextualize_chain to contextualize User's query
    contextualize_q_system_prompt  = """
                                    Given a chat history and the latest user question 
                                    which might reference context in the chat history, formulate a standalone question 
                                    which can be understood without the chat history. Do NOT answer the question, 
                                    just reformulate it if needed and otherwise for generic question such as greetings or compliments just return the original user question. 
                                    """

    contextualize_q_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    # Create decide_chain to determine if it is necessary to trigger Retriever
    decide_system_prompt = """You are an expert that is able to look up information about drugs and supplements. Your name is DoseBuddy \ 
           You have 2 tools, 'RAG' and 'Direct Answer'. Given a chat history and the latest user question, you need to decide what tool to use to respond.\
           You will use the tool 'RAG' if the question pertains to drugs or supplements and you want to look up more information.\
           You will use the tool 'Direct Answer' if the question is generic such as greetings or compliments.\
           Question: {question}
           What tool do you want to use? Only reply 'RAG' or 'Direct Answer'."""

    decide_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", decide_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    ## Assign the output: 'Direct Answer' or 'RAG' to parameter 'topic' / {'topic':'Direct Answer'}
    decide_chain = RunnablePassthrough.assign(
            topic = RunnablePassthrough.assign(question = contextualize_chain ) |decide_prompt | llm_decide | StrOutputParser()
        )

    # Create general_chain to answer general questions without using RAG
    general_system_prompt =  """You are an expert that is able to look up information about drugs and supplements.\
                            Your name is DoseBuddy. Please Respond to the following question:
                            Question: {question}
                            Answer:"""
    
    general_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", general_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    general_chain = general_prompt | llm


    ## In case no prev chat history, directly return user query
    def contextualized_question(info):
        if info.get("chat_history"):
            return contextualize_chain
        else:
            return info["question"]

    # Create retriever_chain

    def format_docs(docs):
        formatted_docs = []
        for doc in docs:
            content_data = doc.page_content
            s3_uri = doc.metadata['id']
            s3_gen_url = generate_presigned_url(s3_uri)
            formatted_doc = f"{content_data}\n\n[More Info]```{s3_gen_url}```"
            formatted_docs.append(formatted_doc)

        # combined_content = "\n\n".join(formatted_docs)
        return formatted_docs
    
    retriever_chain = RunnablePassthrough.assign(
        retrieved_context = contextualized_question | docsearch.as_retriever() | format_docs,
        question = contextualized_question 
    )

    # Create rag_chain

    rag_system_prompt = """You are an expert that is able to look up information about drugs and supplements.\
                       Your name is DoseBuddy. \
                     Based on the following context provided, provide a summarized & concise explanation using a couple of sentences. \
                     Only respond with the information relevant to the user's question \
                     if there are none, make sure you say the `magic words`: 'I don't know, I did not find the relevant data in the knowledge base.' \
                     But you could carry out some conversations with the user to make them feel welcomed and comfortable, in that case you don't have to say the `magic words`. \
                     In the event that there's relevant info, make sure to attach the download button use the following format right after the end of the relevant paragraph: [More Info](`s3_url`) \ 
                     Replace `s3_url` with the URL delimited by triple backticks in the context. \
                     
                     context: {retrieved_context}"""
    
    rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
    )   

    rag_chain = (
        retriever_chain
        | RunnableLambda(inspect_retrieved_context)
        | rag_prompt
        | llm
    )

    # Create the final chain

    def route(info):
        if "rag" in info["topic"].lower():
            return rag_chain
        else:
            return general_chain

    full_chain = decide_chain | RunnableLambda(inspect_topic) | RunnableLambda(
    route
    )

    
    return full_chain.invoke({
        "chat_history": chat_history,
        "question": user_query,
    }).content

def get_gt(user_query, context, chat_history):

    # Create rag_gt_chain

    rag_system_prompt = """You are an expert that is able to look up information about drugs and supplements.\
                       Your name is DoseBuddy. \
                     Based on the following context provided, provide a summarized & concise explanation using a couple of sentences. \
                     Only respond with the information relevant to the user's question \
                     if there are none, make sure you say the `magic words`: 'I don't know, I did not find the relevant data in the knowledge base.' \
                     But you could carry out some conversations with the user to make them feel welcomed and comfortable, in that case you don't have to say the `magic words`. \
                     In the event that there's relevant info, make sure to attach the download button use the following format right after the end of the relevant paragraph: [More Info](`s3_url`) \ 
                     Replace `s3_url` with the URL delimited by triple backticks in the context. \
                     
                     context: {retrieved_context}"""
    
    rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
    ) 


    rag_gt_chain = (
        rag_prompt
        | llm_gt
)

    return rag_gt_chain.invoke({'question':user_query, "chat_history": chat_history,'retrieved_context':context}).content

def ragas_evaluate(user_query,context,answer,ground_truth):
    eval_data = {
        
            "question": [user_query],
            "contexts": [context],
            "answer": [answer],
            "ground_truth": [ground_truth]
        }
    
    dataset_eval = Dataset.from_dict(eval_data)

    result = evaluate(
    dataset_eval,
    metrics=[
        answer_relevancy,
        answer_correctness
    ],
    llm = llm_gt,
    embeddings= OpenAIEmbeddings(model='text-embedding-3-large'),
    raise_exceptions=False
                        ).to_pandas()
    
    return result['answer_relevancy'].values[0], result['answer_correctness'].values[0]


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Dosebuddy. How can I help you?"),
    ]
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            save.append("User: " + message.content)
        elif isinstance(message, AIMessage):
            save.append("Bot:" + message.content)
    st.session_state["stored_session"].append(save)
    st.session_state.chat_history = [AIMessage(content="Hello, I am Dosebuddy. How can I help you?")]

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type="primary")

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session

    


# user input
container2 = st.container()
with container2:
    left, right = st.columns([0.9, 0.1])
    with left:
        user_query_input = st.chat_input("Type your message here...", key = 'Voice_Input')
    with right:
        user_query_audio = speech_to_text(language='en', start_prompt=play_button + microphone,
                            stop_prompt = red_square ,use_container_width=False, just_once=True, key='STT')
    user_query = user_query_input or user_query_audio
    print(user_query)

# user input
download_str = []
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    download_str.append('You: '+ user_query)

    answer = get_answer(user_query, st.session_state.chat_history[-2*K:])

    st.session_state.chat_history.append(AIMessage(content=answer))
    download_str.append('DoseBuddy: ' + answer)

# Keep Conversation History
with st.container(height=600, border=False):
    st.write("Conversation")
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.info(message.content, icon="🧐")
        elif isinstance(message, AIMessage):
            st.success(message.content, icon="🤖")
    
    download_txt = '\n\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_txt,file_name=f"chat_history_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt")