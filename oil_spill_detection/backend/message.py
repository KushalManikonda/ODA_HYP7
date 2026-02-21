import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =========================================================
# ODA MARINE SPILL INTELLIGENCE SYSTEM - Hackathon Version
# =========================================================
# References & Scientific Basis:
# - BOEM Relative Environmental Sensitivity Analysis (RESA): https://espis.boem.gov/final%20reports/5400.pdf
# - NOAA Environmental Sensitivity Index (ESI): https://response.restoration.noaa.gov/oil-and-chemical-spills/oil-spills/resources/environmental-sensitivity-index-esi-maps.html
# - EPA Guidelines for Ecological Risk Assessment (1998): https://www.epa.gov/sites/default/files/2014-11/documents/eco_risk_assessment1998.pdf
# - Species sensitivity: Adapted from Deepwater Horizon NRDA & Pezeshki & DeLaune (2015)
# - Oil persistence modifiers: NOAA & BOEM oil behavior literature
# - OBIS API: https://api.obis.org/
# - Personnel: Illustrative heuristic only – real incidents (e.g. Deepwater Horizon ~47,000 peak responders)

# =========================================================
# Gulf of Mexico Bounding Box
# =========================================================
GULF_MIN_LAT = 18.0
GULF_MAX_LAT = 31.0
GULF_MIN_LON = -98.0
GULF_MAX_LON = -80.0

# =========================================================
# Fetch Marine Biodiversity Data from OBIS
# =========================================================
def fetch_marine_species(lat: float, lon: float, radius_km: float = 100.0) -> list:
    if not (GULF_MIN_LAT <= lat <= GULF_MAX_LAT and GULF_MIN_LON <= lon <= GULF_MAX_LON):
        print("Location is outside Gulf of Mexico bounding box.")
        return []

    url = "https://api.obis.org/v3/occurrence"
    params = {
        "decimalLatitude": lat,
        "decimalLongitude": lon,
        "radius": int(radius_km * 1000),
        "marine": True,
        "size": 3000,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"OBIS API error: {e}")
        return []

    relevant_classes = {
        "Actinopterygii", "Chondrichthyes", "Anthozoa", "Malacostraca",
        "Cephalopoda", "Bivalvia", "Mammalia", "Reptilia"
    }

    species_list = []
    for record in data.get("results", []):
        cls = record.get("class")
        sci_name = record.get("scientificName")
        common_name = record.get("vernacularName", "")
        bathymetry = record.get("bathymetry")

        if (cls in relevant_classes and
            sci_name and
            bathymetry is not None and
            bathymetry < 0):
            species_list.append({
                "scientific": sci_name,
                "common": common_name if common_name else sci_name,
                "class": cls
            })

    return species_list

# =========================================================
# Compute Relative Vulnerability Index (RVI)
# =========================================================
def compute_relative_vulnerability(species_list: list, spill_area_sqkm: float, oil_type: str) -> tuple:
    if not species_list:
        return 0.0, "LOW"

    class_vuln_factors = {
        "Anthozoa": 1.8,
        "Mammalia": 1.7,
        "Reptilia": 1.6,
        "Chondrichthyes": 1.4,
        "Actinopterygii": 1.0,
        "Malacostraca": 0.9,
        "Cephalopoda": 0.9,
        "Bivalvia": 0.8
    }

    class_counts = defaultdict(int)
    for sp in species_list:
        class_counts[sp["class"]] += 1

    vuln_score = sum(count * class_vuln_factors.get(cls, 1.0)
                     for cls, count in class_counts.items())

    MAX_EXPECTED_EXPOSURE = 500.0
    norm_vuln = min(vuln_score / MAX_EXPECTED_EXPOSURE, 1.0)

    MAX_REF_AREA = 100.0
    area_factor = min(spill_area_sqkm / MAX_REF_AREA, 1.0)

    oil_modifiers = {"light": 0.8, "crude": 1.0, "heavy": 1.5}
    oil_factor = oil_modifiers.get(oil_type.lower(), 1.0)

    rvi = (0.6 * norm_vuln + 0.4 * area_factor) * oil_factor * 100 / 1.5
    rvi = max(0.0, min(round(rvi, 2), 100.0))

    if rvi >= 75:
        level = "ECO-CRITICAL"
    elif rvi >= 50:
        level = "HIGH"
    elif rvi >= 25:
        level = "MODERATE"
    else:
        level = "LOW"

    return rvi, level

# =========================================================
# Personnel Prediction (Simplified Heuristic Model)
# =========================================================
def train_personnel_model() -> RandomForestRegressor:
    X = np.array([
        [200,   0],
        [1000,  1],
        [3000,  2],
        [10000, 2],
        [50000, 2]   # reference point (DWH scale)
    ])
    y = np.array([120, 450, 1400, 4200, 47000])

    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model

def predict_personnel(model: RandomForestRegressor, volume_barrels: float, oil_type: str) -> int:
    oil_codes = {"light": 0, "crude": 1, "heavy": 2}
    code = oil_codes.get(oil_type.lower(), 1)
    pred = model.predict([[volume_barrels, code]])[0]
    return int(round(pred))

# =========================================================
# Cleanup Recommendation
# =========================================================
def recommend_cleanup(oil_type: str, risk_level: str) -> str:
    if risk_level in ["HIGH", "ECO-CRITICAL"]:
        return "Mechanical Recovery + Dedicated Wildlife Response Teams (NOAA sensitive habitat protocol)"
    if oil_type.lower() == "heavy":
        return "Containment Booms + Limited/Authorized Dispersants (for persistent oils)"
    return "Skimmers + Sorbent Booms (standard mechanical recovery)"

# =========================================================
# Email Sending Function (Gmail SMTP)
# =========================================================
def send_email_alert(alert_text: str,
                     recipient_email: str,
                     sender_email: str,
                     app_password: str):
    # Clean up any accidental whitespace
    recipient_email = recipient_email.strip()
    sender_email = sender_email.strip()

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "🚨 ODA Marine Spill Intelligence Alert"

    msg.attach(MIMEText(alert_text, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"Alert email successfully sent to {recipient_email}")
    except Exception as e:
        print(f"Email sending failed: {e}")
        print("Common fixes: Check App Password (no spaces), 2FA status, or security alerts in Gmail.")

# =========================================================
# Generate Alert Message
# =========================================================
def generate_alert(
    area_sqkm: float,
    volume_barrels: int,
    oil_type: str,
    lat: float,
    lon: float,
    rvi: float,
    risk_level: str,
    personnel: int,
    cleanup: str
) -> str:
    return f"""\
🚨 ODA MARINE ECOSYSTEM SPILL ALERT 🚨
Simulated Gulf of Mexico Incident
Location: {lat:.4f}°N, {lon:.4f}°W

Spill Characteristics:
• Area: {area_sqkm:.1f} sq km
• Estimated Volume: {volume_barrels:,} barrels
• Oil Type: {oil_type.capitalize()}

Ecological Assessment:
• Relative Vulnerability Index: {rvi}/100
• Risk Level: {risk_level}
High ecological exposure vulnerability detected in regional marine biodiversity

Response Estimate:
• Approximate Personnel Required: {personnel:,} (prototype heuristic – real incidents scale much higher)
• Recommended Cleanup Strategy: {cleanup}

Note: Personnel estimate is illustrative only.
Real-world example: Deepwater Horizon peak response involved ~47,000 people.

Scientific Sources:
• BOEM RESA • NOAA ESI • EPA Ecological Risk Guidelines (1998)
• Deepwater Horizon NRDA studies • OBIS marine biodiversity database
"""

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    # Example spill scenario
    spill_area = 100.0          # sq km
    spill_volume = 3000       # barrels
    oil_type = "heavy"
    latitude = 28.5
    longitude = -89.5

    print("Querying OBIS marine biodiversity database...")
    species_records = fetch_marine_species(latitude, longitude, radius_km=100.0)
    print(f"→ Retrieved {len(species_records)} filtered occurrence records")

    rvi_score, eco_risk_level = compute_relative_vulnerability(species_records, spill_area, oil_type)

    personnel_model = train_personnel_model()
    estimated_personnel = predict_personnel(personnel_model, spill_volume, oil_type)

    cleanup_strategy = recommend_cleanup(oil_type, eco_risk_level)

    alert_text = generate_alert(
        spill_area, spill_volume, oil_type,
        latitude, longitude,
        rvi_score, eco_risk_level,
        estimated_personnel, cleanup_strategy
    )

    print("\n" + "="*80)
    print(alert_text)
    print("="*80)

    # === SEND ALERT VIA EMAIL ===
    # FILL IN / VERIFY THESE VALUES
    YOUR_GMAIL = "priyansh.luckey@gmail.com"
    GMAIL_APP_PASSWORD = "grkzmzyxpuqsqjhv"   # ← PASTE HERE (NO SPACES!)
    RECIPIENT_EMAIL = "1akiraaravind1@gmail.com"           # Change if sending elsewhere

    # Send the alert email
    send_email_alert(
        alert_text=alert_text,
        recipient_email=RECIPIENT_EMAIL,
        sender_email=YOUR_GMAIL,
        app_password=GMAIL_APP_PASSWORD
    )

if __name__ == "__main__":
    main()