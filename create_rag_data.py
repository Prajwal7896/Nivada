import pandas as pd
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("complaints_50000.csv")

# Clean data
df["complaint"] = df["complaint"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip()

# ---------------- CATEGORY → SOLUTION ----------------
CATEGORY_SOLUTIONS = {

    # 🔌 Electricity
    "Power Outage": "Check your main fuse and circuit breaker. If the issue persists, report it to the electricity department.",
    "Low Voltage": "Avoid using heavy appliances and report the issue to the electricity authority.",
    "Frequent Power Cuts": "Report repeated outages to the electricity provider for grid inspection.",
    "Transformer Failure": "Stay away from the area and immediately inform the electricity department.",
    "Street Light Not Working": "Report to municipal or electricity department for repair.",
    "Electric Shock Hazard": "Avoid contact and report immediately to the electricity department.",
    "Meter Fault": "Contact electricity board for meter inspection or replacement.",
    "High Electricity Bill": "Verify usage and request bill correction from electricity provider.",

    # 💧 Water
    "Water Leakage": "Turn off the main valve if possible and report the issue to the water department.",
    "Drinking Water Shortage": "Contact the local water authority for supply updates or tanker services.",
    "Contaminated Water": "Avoid using the water and report immediately for quality testing.",
    "Pipeline Burst": "Report urgently to prevent water loss and damage.",
    "Low Water Pressure": "Check internal plumbing and report to water authority.",
    "Drain Overflow": "Report to municipal department for cleaning.",
    "Sewer Blockage": "Avoid usage and report to sanitation department immediately.",

    # 🚧 Transport
    "Potholes": "Report to road maintenance department for repair.",
    "Road Damage": "Contact municipal authority for inspection and fixing.",
    "Traffic Jam": "Avoid peak hours and report to traffic control department.",
    "Signal Failure": "Report immediately to traffic police.",
    "Parking Issue": "Report illegal parking to authorities.",
    "Accident Hazard": "Inform traffic police and mark the unsafe area.",

    # 🏥 Health
    "Hospital Negligence": "Report to hospital authority or health department.",
    "Ambulance Delay": "Contact emergency helpline and report delay.",
    "Medicine Shortage": "Inform hospital or pharmacy authority.",
    "Doctor Absence": "Report to hospital administration.",
    "Overcharging Hospital": "File complaint with health department.",
    "Poor Hygiene Hospital": "Report sanitation issue to authorities.",

    # 🏙️ Municipal
    "Garbage Collection Issue": "Report to municipal sanitation department.",
    "Street Cleaning": "Request cleaning through municipal services.",
    "Public Toilet Issue": "Report maintenance issue to municipal body.",
    "Stray Animals": "Inform municipal animal control department.",
    "Illegal Construction": "Report to municipal authority.",
    "Water Logging": "Report drainage issue to municipal department.",

    # 🌍 Environment
    "Air Pollution": "Report to environmental control board.",
    "Water Pollution": "Avoid usage and report contamination source.",
    "Noise Pollution": "Report to local authorities or police.",
    "Tree Cutting": "Report to forest department.",
    "Illegal Dumping": "Inform municipal or environmental authority.",

    # 📶 Telecom
    "Internet Down": "Restart router and contact ISP if issue continues.",
    "Slow Internet": "Check usage and contact ISP.",
    "Network": "Try changing location or contact provider.",
    "Call Drop": "Report to telecom provider.",
    "SIM Activation Issue": "Contact telecom support.",
    "Billing Issue": "Verify bill and contact provider.",
    "Recharge Failure": "Check transaction and contact support.",
    "Spam Calls": "Block number and report to telecom provider.",

    # 🚔 Police
    "Theft": "Report immediately to the nearest police station.",
    "Robbery": "Call emergency services and report the incident.",
    "Cyber Fraud": "Report to cyber crime portal immediately.",
    "Online Scam": "Avoid further transactions and report.",
    "Identity Theft": "Secure accounts and report to cyber cell.",
    "Assault": "Seek help and report to police immediately.",
    "Domestic Violence": "Contact helpline and police.",
    "Harassment": "Report to police or relevant authority.",
    "Kidnapping": "Contact emergency police services immediately.",

    # 🔁 DEFAULT
    "Other": "Your complaint has been registered and forwarded to the appropriate department. You will receive updates soon."
}

# ---------------- FUNCTION ----------------
def get_solution(category):
    if not category:
        return CATEGORY_SOLUTIONS["Other"]

    category = category.lower().strip()

    # 🔌 ELECTRICITY
    if any(word in category for word in ["electric", "power", "voltage", "transformer"]):
        return CATEGORY_SOLUTIONS.get("Power Outage")

    if "meter" in category:
        return CATEGORY_SOLUTIONS.get("Meter Fault")

    if "bill" in category:
        return CATEGORY_SOLUTIONS.get("High Electricity Bill")

    # 💧 WATER
    if any(word in category for word in ["water", "pipeline", "leak"]):
        return CATEGORY_SOLUTIONS.get("Water Leakage")

    if "drain" in category or "sewer" in category:
        return CATEGORY_SOLUTIONS.get("Sewer Blockage")

    # 🚧 ROADS
    if "pothole" in category:
        return CATEGORY_SOLUTIONS.get("Potholes")

    if "road" in category:
        return CATEGORY_SOLUTIONS.get("Road Damage")

    if "traffic" in category:
        return CATEGORY_SOLUTIONS.get("Traffic Jam")

    # 📶 TELECOM
    if any(word in category for word in ["network", "internet", "wifi"]):
        return CATEGORY_SOLUTIONS.get("Internet Down")

    if "call drop" in category:
        return CATEGORY_SOLUTIONS.get("Call Drop")

    # 🚔 POLICE
    if any(word in category for word in ["theft", "robbery", "steal"]):
        return CATEGORY_SOLUTIONS.get("Theft")

    # 🔁 DEFAULT
    return CATEGORY_SOLUTIONS["Other"]


# ---------------- BUILD DATASET ----------------
texts = []
solutions = []
categories = []

for _, row in df.iterrows():
    text = row["complaint"]
    category = row["category"]

    solution = get_solution(category)

    texts.append(text)
    solutions.append(solution)
    categories.append(category)

# ---------------- SAVE ----------------
data = {
    "texts": texts,
    "solutions": solutions,
    "categories": categories   # 🔥 extra useful for filtering later
}

with open("complaints_store.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ RAG dataset created successfully (clean + upgraded)")