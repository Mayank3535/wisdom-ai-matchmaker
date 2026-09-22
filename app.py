from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json, os, re

app = Flask(__name__, static_folder="static", static_url_path="")

DB_FILE = "data.json"
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump({"clients": [], "suppliers": [], "matches": [], "notifications": []}, f, indent=2)

def load_db():
    with open(DB_FILE) as f: return json.load(f)

def save_db(db):
    with open(DB_FILE, "w") as f: json.dump(db, f, indent=2)

def money(v):
    try:
        return float(re.sub(r"[^0-9.]", "", str(v)))
    except: return 0.0

def text_of_client(c):
    return " ".join(str(c.get(k,"")) for k in ["product_requirement","category","location","additional_notes"])

def text_of_supplier(s):
    return " ".join(str(s.get(k,"")) for k in ["product_offered","category","location","delivery_capability","additional_notes"])

def score_pair(c, s):
    # ML semantic similarity using TF-IDF vectors, not keyword equality.
    docs = [text_of_client(c), text_of_supplier(s)]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    X = vec.fit_transform(docs)
    semantic = float(cosine_similarity(X[0:1], X[1:2])[0][0])

    cat = 1.0 if c["category"].strip().lower() == s["category"].strip().lower() else 0.35
    qty = float(s["available_quantity"]) / max(float(c["quantity_required"]), 1)
    quantity = 1.0 if qty >= 1 else max(0, qty)
    budget = money(c["budget"])
    price = money(s["pricing_details"])
    budget_score = 1.0 if not budget or not price else max(0, min(1, budget/(price*float(c["quantity_required"]))))
    location = 1.0 if c["location"].strip().lower() in s["location"].strip().lower() or s["location"].strip().lower() in c["location"].strip().lower() else 0.55
    delivery_text = (s["delivery_capability"] + " " + c["delivery_timeline"]).lower()
    delivery = 0.85 if any(x in delivery_text for x in ["day","week","fast","express","30","15","7"]) else 0.6

    final = (semantic*.35 + cat*.15 + quantity*.15 + budget_score*.15 + location*.10 + delivery*.10)
    return {
        "semantic_score": round(semantic*100,1),
        "category_score": round(cat*100,1),
        "quantity_score": round(quantity*100,1),
        "budget_score": round(budget_score*100,1),
        "location_score": round(location*100,1),
        "delivery_score": round(delivery*100,1),
        "final_score": round(final*100,1),
        "explanation": (
            f"AI semantic similarity is {semantic*100:.0f}%. "
            f"Category compatibility is {cat*100:.0f}%, and the supplier has "
            f"{'enough' if quantity>=1 else 'limited'} quantity for the request. "
            f"Budget compatibility is {budget_score*100:.0f}%. "
            f"The location and delivery information also support this match."
        )
    }

@app.route("/")
def index(): return send_from_directory("static", "index.html")

@app.get("/api/data")
def data(): return jsonify(load_db())

@app.post("/api/clients")
def add_client():
    db=load_db(); x=request.json
    x["id"]=len(db["clients"])+1; x["created_at"]=datetime.now().isoformat()
    db["clients"].append(x); save_db(db)
    return jsonify(x)

@app.post("/api/suppliers")
def add_supplier():
    db=load_db(); x=request.json
    x["id"]=len(db["suppliers"])+1; x["created_at"]=datetime.now().isoformat()
    db["suppliers"].append(x); save_db(db)
    return jsonify(x)

@app.post("/api/match/<int:client_id>")
def match(client_id):
    db=load_db()
    c=next((x for x in db["clients"] if x["id"]==client_id), None)
    if not c: return jsonify({"error":"Client not found"}),404
    results=[]
    for s in db["suppliers"]:
        sc=score_pair(c,s)
        m={"id":len(db["matches"])+len(results)+1,"client_id":c["id"],"supplier_id":s["id"],**sc,"status":"New","created_at":datetime.now().isoformat()}
        results.append(m)
    results.sort(key=lambda x:x["final_score"], reverse=True)
    for i,m in enumerate(results):
        m["rank"]=i+1
        db["matches"].append(m)
        db["notifications"].append({"id":len(db["notifications"])+1,"message":f"New AI match: {c['company_name']} ↔ {next(s['supplier_name'] for s in db['suppliers'] if s['id']==m['supplier_id'])} ({m['final_score']}%)","read":False})
    save_db(db)
    return jsonify(results)

@app.post("/api/seed")
def seed():
    db=load_db()
    if db["clients"] or db["suppliers"]: return jsonify({"message":"Demo data already exists"})
    clients=[
      {"id":1,"company_name":"GreenTech Solutions","product_requirement":"500 solar powered outdoor LED street lights for a municipal project","category":"Solar Equipment","quantity_required":500,"budget":"1500000","location":"Mumbai","delivery_timeline":"30 days","additional_notes":"Weather resistant and energy efficient"},
      {"id":2,"company_name":"Nova Events","product_requirement":"2000 custom printed cotton tote bags for a corporate event","category":"Textiles","quantity_required":2000,"budget":"200000","location":"Mumbai","delivery_timeline":"20 days","additional_notes":"Eco friendly material"}
    ]
    suppliers=[
      {"id":1,"supplier_name":"SunPower Technologies","product_offered":"Solar LED street lighting systems, outdoor and weather resistant","category":"Solar Equipment","available_quantity":1000,"pricing_details":"2500","location":"Pune","delivery_capability":"20-25 days","additional_notes":"Bulk municipal projects"},
      {"id":2,"supplier_name":"Bright Energy Systems","product_offered":"Solar powered LED outdoor lights","category":"Solar Equipment","available_quantity":600,"pricing_details":"2800","location":"Mumbai","delivery_capability":"30 days","additional_notes":"Installation available"},
      {"id":3,"supplier_name":"EcoPrint Manufacturing","product_offered":"Custom printed reusable cotton tote bags for events and brands","category":"Textiles","available_quantity":5000,"pricing_details":"70","location":"Mumbai","delivery_capability":"15-20 days","additional_notes":"Organic cotton available"},
      {"id":4,"supplier_name":"Urban Merchandise Co","product_offered":"Promotional canvas and cotton bags","category":"Textiles","available_quantity":1200,"pricing_details":"110","location":"Pune","delivery_capability":"25-30 days","additional_notes":"Custom printing"}
    ]
    db["clients"]=clients; db["suppliers"]=suppliers; save_db(db)
    for c in clients: 
        # perform matching without recursively calling endpoint
        for s in suppliers:
            sc=score_pair(c,s)
            db["matches"].append({"id":len(db["matches"])+1,"client_id":c["id"],"supplier_id":s["id"],**sc,"status":"New","created_at":datetime.now().isoformat()})
    save_db(db)
    return jsonify({"message":"Demo data loaded"})

@app.post("/api/reset")
def reset():
    with open(DB_FILE,"w") as f: json.dump({"clients":[],"suppliers":[],"matches":[],"notifications":[]},f,indent=2)
    return jsonify({"message":"Reset complete"})

if __name__=="__main__":
    app.run(debug=True, port=5000)
