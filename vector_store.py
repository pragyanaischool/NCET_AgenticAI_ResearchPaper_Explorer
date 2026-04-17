from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)
    vector_db = FAISS.from_texts(chunks, embedding_model)

    return vector_db


def retrieve_context(vector_db, query, k=5):
    docs = vector_db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
