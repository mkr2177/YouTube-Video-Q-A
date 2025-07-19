import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="YouTube Video Q&A", page_icon="🎥")

if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "page" not in st.session_state:
    st.session_state.page = "video_input"
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "main_chain" not in st.session_state:
    st.session_state.main_chain = None

def go_to_qa_page():
    if st.session_state.video_id.strip():
        try:
            # Try to get the best available transcript (any language)
            transcript_list = YouTubeTranscriptApi.list_transcripts(st.session_state.video_id)

            # Try manual first, then fallback to auto-generated
            try:
                transcript = transcript_list.find_manually_created_transcript(transcript_list._manually_created_transcripts.keys())
            except:
                transcript = transcript_list.find_generated_transcript(transcript_list._generated_transcripts.keys())

            transcript_text = transcript.fetch()
            transcript_text_combined = " ".join(chunk.text for chunk in transcript_text)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript_text_combined])
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            llm = ChatOpenAI(
                model="llama3-8b-8192",
                openai_api_key=f"{YOUR GROQ API KEY}",
                openai_api_base="https://api.groq.com/openai/v1"
            )

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=["context", "question"]
            )

            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })

            parser = StrOutputParser()

            main_chain = parallel_chain | prompt | llm | parser

            st.session_state.vector_store = vector_store
            st.session_state.main_chain = main_chain
            st.session_state.page = "qa_interface"

        except TranscriptsDisabled:
            st.error("❌ No captions available for this video.")
        except Exception as e:
            st.error(f"⚠️ Error occurred: {e}")

def go_back():
    st.session_state.page = "video_input"

# --- Page 1: Enter Video ID ---
if st.session_state.page == "video_input":
    st.title("🎥 YouTube Video Q&A")
    st.text_input("Enter YouTube Video ID:", key="video_id")
    st.button("Submit", on_click=go_to_qa_page)

# --- Page 2: Ask a Question ---
elif st.session_state.page == "qa_interface":
    st.title("❓ Ask a Question About the Video")

    video_url = f"https://www.youtube.com/watch?v={st.session_state.video_id}"
    st.markdown(f"[🔗 Open Video]({video_url})", unsafe_allow_html=True)

    question = st.text_input("Enter your question about the video:")
    if st.button("Get Answer") and question:
        try:
            answer = st.session_state.main_chain.invoke(question)
            st.success(f"🤖 Answer: {answer}")
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")

    st.button("🔙 Back", on_click=go_back)
