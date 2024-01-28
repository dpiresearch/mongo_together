# Mongo and Together ai RAG implementation

This is based off of this tutorial https://www.together.ai/blog/rag-tutorial-mongodb  with some cleanup

## Tutorial code

start.py - This creates the embeddings, the search index configured in MongodB takes over
retrieve.py - This retrieves the top 10 entries that match the query embeddings and serves up the most likely answer.
