import together
import pymongo
import os
from tqdm import tqdm
from typing import List

def generate_embedding(input_texts: List[str], model_api_string: str) -> List[List[float]]:
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

TOGETHER_API_KEY=os.environ.get("TOGETHER_API_KEY")
MONGODB_URI=os.environ.get("MONGODB_URI")

client = pymongo.MongoClient(MONGODB_URI)

db = client.sample_airbnb
collection_airbnb = db.listingsAndReviews
embedding_model_string = 'togethercomputer/m2-bert-80M-8k-retrieval' # model API string from Together.
vector_database_field_name = 'embedding_together_m2-bert-8k-retrieval' # define your embedding field name.
NUM_DOC_LIMIT = 200 # the number of documents you will process and generate embeddings.

# Example query.
query = "apartment with a great view near a coast or beach for 4 people"
query_emb = generate_embedding([query], embedding_model_string)[0]

results = collection_airbnb.aggregate([
  {
    "$vectorSearch": {
      "queryVector": query_emb,
      "path": vector_database_field_name,
      "numCandidates": 100, # this should be 10-20x the limit
      "limit": 10, # the number of documents to return in the results
      "index": "SemanticSearch", # the index name you used in Step 4.
    }
  }
])
results_as_dict = {doc['name']: doc for doc in results}

print(f"From your query \"{query}\", the following airbnb listings were found:\n")
print("\n".join([str(i+1) + ". " + name for (i, name) in enumerate(results_as_dict.keys())]))

your_task_prompt = (
    "From the given airbnb listing data, choose an apartment with a great view near a coast or beach for 4 people to stay for 4 nights. "
    "I want the apartment to have easy access to good restaurants. "
    "Tell me the name of the listing and why this works for me."
)


listing_data = ""
keys_to_extract = ["name", "summary", "space", "description", "neighborhood_overview", "notes", "transit", "access", "interaction", "house_rules", "property_type", "room_type", "bed_type", "minimum_nights", "maximum_nights", "accommodates", "bedrooms", "beds"]

# model_api_string = 'mistralai/Mistral-7B-Instruct-v0.2'
model_api_string = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
stop_sequences=['\n\n']

for doc in results_as_dict.values():
  listing_data += f"Listing name: {doc['name']}\n"
  for (k, v) in doc.items():
    if not(k in keys_to_extract) or ("embedding" in k): continue
    if k == "name": continue
    listing_data += k + ": " + str(v) + "\n"
  listing_data += "\n"

augmented_prompt = (
    "airbnb listing data:\n"
    f"{listing_data}\n\n"
    f"{your_task_prompt}\n\n"
)

def call_together_completion(augmented_prompt, model_api_string, stop_sequences):
    response = together.Complete.create(
        prompt=augmented_prompt,
        model=model_api_string,
        max_tokens = 512,
        temperature = 0.8,
        top_k = 60,
        top_p = 0.6,
        repetition_penalty = 1.1,
        stop = stop_sequences,
        )
    print("response")
    # print("XXX:"+response)
    print(response["output"]["choices"][0]["text"])

# call_together_completion(augmented_prompt, model_api_string, stop_sequences)

messages = [
    {
        "role": "system",
        "content": f"Airbnb listing data: {listing_data}"
    },
    {
        "role": "user",
        "content": f"{query}"
    }
]

import openai

client = openai.OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
    )
chat_completion = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=messages,
    temperature=0.7,
    max_tokens=1024,
)
response = chat_completion.choices[0].message.content
print("Together openai response:\n", response)


