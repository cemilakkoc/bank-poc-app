from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
import transformers
import json
import os
import torch

local_transformers_model = os.environ.get('LOCAL_TRANSFORMERS_MODEL', False)
model = SentenceTransformer(
    'paraphrase-multilingual-mpnet-base-v2' if not local_transformers_model else './transformers-model')

app = Flask(__name__)

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

if os.path.exists(model_id):
    model_id = "./meta-llama/Meta-Llama-3-70B-Instruct"

gpu_available = torch.cuda.is_available()

kwargs = {
    "model_kwargs": {"torch_dtype": torch.bfloat16},
} if gpu_available else {}

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    # if gpu is available, use it
    device="auto" if gpu_available else -1,
    **kwargs
)

# get all intents and its examples
categories = json.load(open('categories.json'))
intents = {}

for parent_name in categories:
    subcategories = categories[parent_name]
    for subcategory in subcategories:
        name = subcategory['subcategory_name']
        catid = subcategory['subcategory_id']
        examples = subcategory['description']

        # embed all examples
        embeddings = model.encode(examples)
        intents[name + "-" + catid] = {
            'id': catid,
            'name': name,
            'examples': examples,
            'embeddings': embeddings,
            'parent': parent_name
        }


def get_closest_intent(query, top_k=20):
    query_embedding = model.encode(query, precision="float32")
    scores = {}
    intent_str = ""

    # top k from each parent category
    for parent_name in categories:
        subcategories = categories[parent_name]
        for subcategory in subcategories:
            name = subcategory['subcategory_name']
            catid = subcategory['subcategory_id']
            examples = subcategory['description']
            embeddings = model.encode(examples)
            catname = name + "-" + catid
            parent = parent_name
            scores[catname] = util.pytorch_cos_sim(
                query_embedding, embeddings).item()

    # sort by score, highest first
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    names = []
    for i in range(top_k):
        names.append(sorted_scores[i][0])

    # sort by score, highest first
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    names = []
    for i in range(top_k):
        names.append(sorted_scores[i][0])
    _intents = [intents[name] for name in names]
    # sort by parent category, parent id is the first 2 digits of the id
    _intents.sort(key=lambda x: x['id'])

    parent_categories = {}

    for intent in _intents:
        parent = intent['parent']
        if parent not in parent_categories:
            parent_categories[parent] = []
        parent_categories[parent].append(intent['name'] + "-" + intent['id'])

    for idx, parent in enumerate(parent_categories):
        # intent_str += f"{idx + 1}. **{parent} [ID: {intents[parent_categories[parent][0]]['id'][0:2]}]:**\n"
        intent_str += f"{idx + 1}. **{parent}:**\n"
        for intent in parent_categories[parent]:
            intent_str += f"  - **{intent.split('-')[0]} [ID: {intents[intent]['id']}]:**: {intents[intent]['examples']}\n"

    return intent_str


def ask_llm(query, k=10):
    #  check if the model is loaded
    if not pipeline.model:
        return jsonify({"error": "Model not loaded yet. Please try again later."})

    intents = get_closest_intent(query, k)

    prompt = f"""
    בהתחשב בקטגוריות ודוגמאות הבאות, סווג את הקלט של המשתמש לתחום ותת-תחום המתאימים ביותר. הקטגוריות מוגדרות כדלקמן:

    תחומים ותת-תחומים:

    {intents}

    הפלט צריך להיות בפורמט הבא:

[BEGIN]SubcategoryID,SubcategoryID,SubcategoryID[END]

    ספק את שלושת הכוונות האפשריות בעלות הביטחון הגבוה ביותר.

    דוגמאות:
    קלט: "אני צריך הלוואה דחופה"
    [BEGIN]10010,10020,10030[END]

    קלט: "מה היתרה בחשבון שלי?"
    [BEGIN]20020,20021,20030[END]

    קלט: "אני רוצה להזמין כרטיס אשראי חדש"
    [BEGIN]30030,30031,30032[END]
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"User Input: {query}"},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0,
        top_p=0.9,
    )

    response_text = outputs[0]["generated_text"][-1]["content"]
    response_text = response_text.strip()

    #  get between [BEGIN] and [END]
    if "[BEGIN]" in response_text and "[END]" in response_text:
        response_text = response_text.split("[BEGIN]")[1].split("[END]")[0]
        response_text = response_text.strip()

    #  get possible intents
    possible_intents = response_text.split(",")
    possible_intents = [
        str(intent).strip() for intent in possible_intents if intent.isdigit()]

    #  get intent names, subcategory_name by subcategory_id
    intent_names = []
    for intent in possible_intents:
        for parent_name in categories:
            subcategories = categories[parent_name]
            for subcategory in subcategories:
                if str(subcategory['subcategory_id']) == str(intent):
                    intent_names.append({
                        "parent": parent_name,
                        "name": subcategory['subcategory_name'],
                        "id": subcategory['subcategory_id']
                    })
                    break

    return intent_names


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    limit = data.get('limit', 5)
    user_prompt = data.get('user_prompt', "אני רוצה לדבר עם בנקאי")

    _intents = ask_llm(user_prompt, k=limit)

    return jsonify({"response": _intents})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
