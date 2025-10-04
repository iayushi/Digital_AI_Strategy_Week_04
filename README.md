# Digital AI Strategy - Week 4

## Course Overview

Welcome to the **Digital AI Strategy** course! This repository contains materials and resources for **Week 4**, focusing on **Industry Competition Analysis, Generative AI Strategy, and Digital Strategy**.

### Core Topics This Week:

- **Generative AI (GenAI) Adoption Levels** and their competitive implications.
- The **Forces that Shape Industry Competition** (e.g., Porter's Five Forces).
- Applying the **Porter Framework** to formulate **Digital, AI, and GenAI Strategy**.
- Identifying strategic responses to technology-driven industry disruption.

### What's Included

- **DAIS_Week4_RAG.py**: Interactive Streamlit application with RAG (Retrieval Augmented Generation) capabilities (Note: Filename updated for clarity)
- **Course Materials**: Vector database and supporting resources specific to Week 4 content.

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run DAIS_Week4_RAG.py`
3. Enter your API key for your preferred LLM provider
4. Start asking questions about Industry Competition, Porter's, and GenAI Strategy concepts!

## 🗄️ Vector Database Implementation

### Current Implementation Overview (Updated for Week 4)

The application uses the following vector database setup:

- **Embedding Model**: `all-MiniLM-L6-v2` (HuggingFace Sentence Transformers)
- **Vector Store**: ChromaDB with local persistence
- **Storage Location**: `./Week_4_Oct2025/` directory
- **Retrieval Method**: Similarity search with top-k results (k=5)

### Setting Up and Configuring the Vector Database

#### Here we are Using the Pre-built Database

The repository includes a pre-built vector database in the `Week_4_Oct2025` directory. To use it:

1. Ensure the directory exists and contains the ChromaDB files
2. Update the `PERSIST_DIRECTORY` variable in `DAIS_Week4_RAG.py`:
   ```python
   PERSIST_DIRECTORY = "./Week_4_Oct2025"


### How Vector Embeddings Work

Vector embeddings transform text content into numerical representations (vectors) that capture semantic meaning. Similar concepts end up close together in vector space, enabling:

1. **Semantic similarity matching** - Find related concepts even with different wording
2. **Context-aware retrieval** - Understand the meaning behind queries
3. **Efficient similarity search** - Quickly find relevant course materials

### Current Implementation Overview

The application uses the following vector database setup:

- **Embedding Model**: `all-MiniLM-L6-v2` (HuggingFace Sentence Transformers)
- **Vector Store**: ChromaDB with local persistence
- **Storage Location**: `./Week_1_31Aug2025/` directory
- **Retrieval Method**: Similarity search with top-k results (k=5)

### Step-by-Step Guide: Generating Vector Embeddings

#### 1. Prepare Your Course Content

```python
# Example: Prepare documents for embedding
from langchain.schema import Document

# Your course materials as text chunks
course_materials = [
    "Information Systems form the backbone of digital operations...",
    "Digital Platforms create ecosystems for digital interactions...",
    "Artificial Intelligence enhances decision-making capabilities...",
    # Add more content
]

# Convert to Document objects
documents = [
    Document(page_content=content, metadata={"source": f"chunk_{i}"})
    for i, content in enumerate(course_materials)
]
```

#### 2. Initialize the Embedding Model

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the same embedding model used in the application
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': True}
)
```

#### 3. Create and Populate the Vector Database

```python
from langchain_community.vectorstores import Chroma

# Create new vector database
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./your_course_vectordb"
)

# Persist the database to disk
vectorstore.persist()
```

#### 4. Test the Vector Database

```python
# Test similarity search
query = "What are digital platforms?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content[:100]}...")
```


#### Option 2: Creating Your Own Database

1. **Prepare your content**: Organize course materials into text chunks
2. **Install required packages**: Ensure you have all dependencies from `requirements.txt`
3. **Create the database**:

```python
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3  # Required for ChromaDB compatibility

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Your implementation here
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load your documents
documents = [...]  # Your course content as Document objects

# Create vectorstore
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./your_custom_vectordb"
)
```

4. **Update the application**: Modify `PERSIST_DIRECTORY` to point to your new database

### Configuration Options and Customization

#### Embedding Model Options

You can customize the embedding model based on your needs:

```python
# Option 1: Multilingual support
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Option 2: Higher accuracy (larger model)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Option 3: Domain-specific models
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/allenai-specter"  # Scientific papers
)
```

#### Search Configuration

Customize the retrieval behavior:

```python
# In the run_similarity_search function
def run_similarity_search(query):
    # Adjust k for more/fewer results
    results = vectorstore.similarity_search(
        query, 
        k=10,  # Return top 10 instead of 5
        filter={"source": "specific_topic"}  # Optional filtering
    )
    return results

# Alternative: Use similarity search with score threshold
def run_similarity_search_with_score(query):
    results = vectorstore.similarity_search_with_score(
        query, 
        k=5,
        score_threshold=0.7  # Only return results above similarity threshold
    )
    return [doc for doc, score in results if score > 0.7]
```

#### Database Configuration

Customize ChromaDB settings:

```python
# Advanced ChromaDB configuration
vectorstore = Chroma(
    persist_directory="./custom_vectordb",
    embedding_function=embedding_model,
    collection_name="course_materials",  # Custom collection name
    collection_metadata={"description": "Digital AI Strategy Course Content"}
)
```

### Examples of Extending Functionality

#### 1. Adding New Course Materials

```python
# Add new documents to existing database
new_documents = [
    Document(page_content="New course content...", metadata={"week": "2"})
]

vectorstore.add_documents(new_documents)
vectorstore.persist()
```

#### 2. Multi-Collection Setup

```python
# Create separate collections for different course weeks
week1_vectorstore = Chroma(
    persist_directory="./vectordb",
    embedding_function=embedding_model,
    collection_name="week1_materials"
)

week2_vectorstore = Chroma(
    persist_directory="./vectordb",
    embedding_function=embedding_model,
    collection_name="week2_materials"
)
```

#### 3. Metadata Filtering

```python
# Search with metadata filters
def search_by_topic(query, topic):
    results = vectorstore.similarity_search(
        query,
        k=5,
        filter={"topic": topic}
    )
    return results

# Usage
ai_results = search_by_topic("machine learning", "AI")
strategy_results = search_by_topic("planning", "Business Strategy")
```

#### 4. Hybrid Search (Vector + Keyword)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Combine vector similarity with keyword search
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
keyword_retriever = BM25Retriever.from_documents(documents)

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.7, 0.3]  # 70% vector, 30% keyword
)
```

#### 5. Custom Similarity Metrics

```python
# Use different distance metrics
vectorstore = Chroma(
    persist_directory="./vectordb",
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  # Options: l2, ip, cosine
)
```

### Performance Optimization Tips

1. **Batch Processing**: Process documents in batches for better performance
2. **GPU Acceleration**: Use CUDA for embedding generation if available
3. **Chunking Strategy**: Optimize text chunk size (typically 200-1000 tokens)
4. **Caching**: Cache embeddings to avoid recomputation
5. **Index Optimization**: Tune ChromaDB parameters for your use case

### Troubleshooting Common Issues

#### Issue 1: SQLite3 Compatibility
```python
# Fix for ChromaDB SQLite issues
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
```

#### Issue 2: CUDA/GPU Issues
```python
# Force CPU usage if GPU issues
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

#### Issue 3: Memory Issues with Large Datasets
```python
# Process in smaller batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vectorstore.add_documents(batch)
    vectorstore.persist()
```

### 🔧 Helper Script: create_embeddings.py

The repository includes a powerful helper script that simplifies creating vector embeddings from your course content. This script allows you to easily convert .md or .txt files into vector embeddings and store them in a ChromaDB database.

#### Features

- **Multiple File Support**: Process multiple .md or .txt files at once
- **Customizable Chunking**: Adjust chunk sizes and overlap for optimal performance
- **GPU/CPU Support**: Automatically detect and use GPU acceleration when available
- **Database Management**: Create new databases or update existing ones
- **Error Handling**: Comprehensive error handling with user-friendly feedback
- **Testing**: Built-in database testing with sample queries

#### Basic Usage

```bash
# Create embeddings from a single file
python create_embeddings.py course_material.md

# Process multiple files
python create_embeddings.py file1.md file2.txt file3.md

# Process all markdown files in a directory
python create_embeddings.py *.md
```

#### Advanced Usage

```bash
# Use custom chunk size and enable GPU
python create_embeddings.py --chunk-size 1000 --use-gpu files/*.md

# Specify custom database location
python create_embeddings.py --db-path ./my_vectordb files/*.txt

# Use a different embedding model
python create_embeddings.py --model sentence-transformers/all-mpnet-base-v2 *.md

# Create with custom collection name
python create_embeddings.py --collection-name "week2_materials" week2/*.md
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model name for embeddings | `all-MiniLM-L6-v2` |
| `--chunk-size` | Size of text chunks for processing | `500` |
| `--chunk-overlap` | Overlap between text chunks | `50` |
| `--use-gpu` | Use GPU for embedding generation if available | `false` |
| `--db-path` | Directory to store the vector database | `./course_vectordb` |
| `--collection-name` | Name of the vector database collection | `course_materials` |
| `--test-query` | Query to test the database with | `"digital strategy"` |
| `--no-test` | Skip testing the vector database after creation | `false` |

#### Integration with DAIS_Week1_RAG.py

After creating your vector database with the helper script, update the `PERSIST_DIRECTORY` variable in `DAIS_Week1_RAG.py`:

```python
# Update this line in DAIS_Week1_RAG.py
PERSIST_DIRECTORY = "./your_custom_vectordb"  # Path from --db-path option
```

#### Example Workflow

1. **Prepare your content**: Organize course materials into .md or .txt files
2. **Create embeddings**: Use the helper script to process your files
3. **Update the app**: Point the Streamlit app to your new database
4. **Test**: Run the app and verify that your content is searchable

```bash
# Step 1: Create embeddings from your course materials
python create_embeddings.py --chunk-size 800 --use-gpu course_materials/*.md

# Step 2: Update DAIS_Week1_RAG.py to use your database
# PERSIST_DIRECTORY = "./course_vectordb"

# Step 3: Run the Streamlit app
streamlit run DAIS_Week1_RAG.py
```

#### Troubleshooting the Helper Script

**Issue 1: Model Download Errors**
```bash
# Ensure you have internet connectivity for first-time model download
# Models are cached locally after first download
```

**Issue 2: Memory Issues with Large Files**
```bash
# Use smaller chunk sizes for large files
python create_embeddings.py --chunk-size 300 large_file.md
```

**Issue 3: GPU Issues**
```bash
# Force CPU usage if GPU issues occur
python create_embeddings.py large_file.md  # GPU detection is automatic
```

### Best Practices for Course Material Implementation

1. **Content Preparation**:
   - Clean and preprocess text content
   - Maintain consistent formatting
   - Include relevant metadata (topics, weeks, difficulty level)

2. **Chunking Strategy**:
   - Split content into logical, coherent chunks
   - Overlap chunks slightly to maintain context
   - Size chunks appropriately (aim for 200-500 tokens)

3. **Database Management**:
   - Regular backups of vector databases
   - Version control for database schemas
   - Monitor database size and performance

4. **Quality Assurance**:
   - Test retrieval quality with sample queries
   - Validate that similar concepts are found together
   - Monitor and improve relevance over time

## Sample Questions to Explore Course Concepts

Sample Questions to Explore Course Concepts
This collection of sample questions is designed to help you chat with this bot to understand the Week 4 content on Industry Competition and GenAI Strategy through relatable, practical, and creative analogies.

🦸‍♂️ Pop Culture & Analogies
GenAI and Competitive Dynamics
"🌌 Thanos and GenAI Power: If a company gets a monopoly on a key GenAI model (like Thanos getting an Infinity Stone), which of Porter's Five Forces becomes instantly overwhelming for competitors, and why?"

"♟️ Grand Theft Strategy: Compare the Level of adoption of GenAI to different vehicles in Grand Theft Auto (bicycle, car, jet). How does upgrading your 'vehicle' affect your competitive escape speed and ability to navigate the 'city' (market)?"

👶 Simple Explanations
Explaining Core Concepts
"🧒 Explain Porter's Rivalry to a 5-year-old: Use the analogy of kids sharing toys (or not sharing!). How does a new, amazing toy (like GenAI) change the 'sharing' behavior (rivalry) among the kids (companies)?"

"💡 GenAI Adoption Levels for Dummies: Use the analogy of cooking a meal. Explain the difference between 'Level 1: Novice' and 'Level 4: Master Chef' GenAI adoption in terms of tools used (knife vs. advanced oven) and the quality of the final 'dish' (business outcome)."

📝 Class Preparation: Cold Call Practice
Prepare for Challenging Strategy Questions
"🛑 Cold Call: Porter's on Digital Strategy: You are cold-called: 'Identify a core principle of digital strategy that either strengthens or weakens the Threat of New Entrants, and justify your answer using specific examples from platforms like Netflix or Airbnb.'"

"🤯 Cold Call: GenAI and Supplier Power: You are cold-called: 'Discuss whether the rapid proliferation of open-source Generative AI models is likely to increase or decrease the Bargaining Power of Suppliers (e.g., cloud providers like AWS/Azure) in the long run. Why?'"

🎯 Application-Focused Questions
Deepening the Core Topics
"📈 Adoption Impact: What is the critical difference between Level 2 (Experimentation) and Level 3 (Integration) of GenAI adoption, and what organizational challenges must a company overcome to make that jump?"

"🔄 Industry Forces in FinTech: Apply the Threat of Substitutes force to the Financial Services industry. How does a GenAI-powered financial advisor (a substitute) change the competitive landscape for traditional human financial advisors?"


## How to Use These Questions

1. **Choose your favorite analogies**: Start with contexts you're most familiar with
2. **Layer the complexity**: Begin with simple comparisons and add details
3. **Mix and match**: Combine different analogies for richer understanding
4. **Create your own**: Use these as inspiration for personal analogies
5. **Test with the RAG system**: Ask these questions in the course application
6. **Discuss with peers**: Share analogies and learn from others' perspectives

## Tips for Creating Your Own Analogies

- **Start with what you know**: Use your hobbies, interests, and experiences
- **Focus on relationships**: How do components interact and depend on each other?
- **Consider scale**: How do systems grow and evolve over time?
- **Think about problems**: What challenges exist and how are they solved?
- **Explore benefits**: What value is created and for whom?

Remember: The goal is to make complex concepts accessible and memorable. The best analogies are the ones that resonate with your personal experience and help you see familiar patterns in new contexts!*
