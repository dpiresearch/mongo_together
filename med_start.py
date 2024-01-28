import together
import pymongo
import os
from tqdm import tqdm

TOGETHER_API_KEY=os.environ.get("TOGETHER_API_KEY")
MONGODB_URI=os.environ.get("MONGODB_URI")

client = pymongo.MongoClient(MONGODB_URI)

from typing import List

def generate_embeddings(input_texts: List[str], model_api_string: str) -> List[List[float]]:
    """Generate embeddings from Together python library.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
    """
    together_client = together.Together()
    outputs = together_client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    )
    return [x.embedding for x in outputs.data]

embedding_model_string = 'togethercomputer/m2-bert-80M-8k-retrieval' # model API string from Together.
model_api_string = 'togethercomputer/m2-bert-80M-8k-retrieval' # model API string from Together.
vector_database_field_name = 'embedding_together_m2-bert-8k-retrieval' # define your embedding field name.
NUM_DOC_LIMIT = 200 # the number of documents you will process and generate embeddings.

# sample_output = generate_embeddings(["This is a test."], embedding_model_string)
# print(f"Embedding size is: {str(len(sample_output[0]))}")

db = client.MedCluster
collection_med= db.MedDB.MedCollection
print(collection_med)

keys_to_extract = ["Prompt", "Completion"]

for doc in tqdm(collection_med.find({"Completion":{"$exists": True}}).limit(NUM_DOC_LIMIT), desc="Document Processing "):
  extracted_str = "\n".join([k + ": " + str(doc[k]) for k in keys_to_extract if k in doc])
  if vector_database_field_name not in doc:
    doc[vector_database_field_name] = generate_embeddings([extracted_str], embedding_model_string)[0]
  collection_med.replace_one({'_id': doc['_id']}, doc)

