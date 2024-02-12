
from fastapi import FastAPI
# from langchain.evaluation import load_evaluator
# from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel
import requests




app = FastAPI()
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": "Bearer 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"}


class answers(BaseModel):
    student_ans: str
    reference_ans: str

# embedding_model = HuggingFaceEmbeddings()
# evaluator = load_evaluator("embedding_distance", embeddings=embedding_model)

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	

@app.get("/")
def info():
    return {"deatiles":"this is a apii"}

@app.post("/p")
async def similarity_endpoint(item:answers):
    pre = item.student_ans
    ref = item.reference_ans
    
    # result = evaluator.evaluate_strings(prediction="g", reference="g")
    output = query({
	"inputs": {
		"source_sentence": ref,
		"sentences": [
			pre
		]
	},
})
    return {"result":output}





