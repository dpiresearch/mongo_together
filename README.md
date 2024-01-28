# Mongo and Together ai RAG implementation

This is based off of this tutorial https://www.together.ai/blog/rag-tutorial-mongodb  with some cleanup

## Tutorial code

start.py - This creates the embeddings, the search index configured in MongodB takes over

retrieve.py - This retrieves the top 10 entries that match the query embeddings and serves up the most likely answer.

## Hackathon code

### Pre-requisites
Upload the data into MongoDB using MongoImport https://www.mongodb.com/docs/database-tools/mongoimport/

med_start.py - This creates the embeddings for the data uploaded using together ai embeddings

med_retrieve.py - This creates embeddings for the query and looks up the most likely answer in MongoDB and asks together ai to pick the most likely answer.
