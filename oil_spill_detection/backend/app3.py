# app3.py  – Oil Spill Detection with Email Alerts & PDF Reports
# -----------------------------------------------------------------
# Changes over app1.py:
#   • Signup/login now stores & reads user email
#   • message.py logic integrated: after every analysis an eco-intelligence
#     alert is built and emailed to the signed-in user
#   • calculate_volume also generates a PDF report + emails it
# -----------------------------------------------------------------

import pymysql
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, send_file, g, jsonify)
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')          # non-interactive backend – safe for threads
import matplotlib.pyplot as plt
import scipy.io
import os
import uuid
import threading
import smtplib
import io
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from PIL import Image
from roboflow import Roboflow

from fpdf import FPDF
from twilio.rest import Client as TwilioClient

# =============================================================================
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# In-memory task store  { task_id: {status, area, ...} }
tasks = {}

# =============================================================================
# Database
# =============================================================================
DB_HOST     = 'localhost'
DB_USER     = 'root'
DB_PASSWORD = '1602-23-737-002'
DB_NAME     = 'oil_spill_db'
DB_PORT     = 3306

def get_db():
    if 'db' not in g:
        g.db = pymysql.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD,
            database=DB_NAME, port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor
        )
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# =============================================================================
# Email credentials  (sender = the ODA system account)
# =============================================================================
SENDER_EMAIL    = "priyansh.luckey@gmail.com"
SENDER_APP_PASS = "grkzmzyxpuqsqjhv"

# =============================================================================
# Twilio SMS credentials
# =============================================================================
TWILIO_ACCOUNT_SID  = "AC822e2b410e58ea3cc8631e91e50b0ae4"
TWILIO_AUTH_TOKEN   = "0bf7f4b7c18144900363d1a8cf40393e"
TWILIO_PHONE_NUMBER = "+18028108515"   # Twilio sender number

CONFIDENCE_THRESHOLD = 0.85


def _sms_severity(area_sqkm: float, rvi: float) -> float:
    """Simple severity score 0-1 used to decide SMS content priority."""
    normalized_area = min(area_sqkm / 10.0, 1.0)
    normalized_rvi  = min(rvi / 100.0, 1.0)
    return round(0.6 * normalized_rvi + 0.4 * normalized_area, 2)


def send_sms_alert(to_phone: str, area_sqm: float, oil_type: str,
                  rvi: float, risk_level: str, personnel: int,
                  cleanup: str, confidence: float = 1.0):
    """Send a Twilio SMS alert. Silently skips if phone is blank."""
    if not to_phone or not to_phone.strip():
        return
    # Normalize to E.164: strip whitespace/dashes/parens, ensure leading +
    phone = re.sub(r'[\s\-\(\)\.]', '', to_phone.strip())
    if not phone.startswith('+'):
        phone = '+' + phone
    # Basic sanity: must be + followed by 7-15 digits
    if not re.fullmatch(r'\+[1-9]\d{6,14}', phone):
        print(f"[sms] Invalid phone number '{phone}' – skipped. Must be E.164 e.g. +919959583328")
        return
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"[sms] Confidence {confidence:.2f} below threshold – skipped.")
        return
    area_sqkm = area_sqm / 1e6
    severity  = _sms_severity(area_sqkm, rvi)
    body = (
        f"OIL SPILL ALERT\n"
        f"Area: {area_sqkm:.4f} sq km\n"
        f"Oil Type: {oil_type.capitalize()}\n"
        f"Confidence: {confidence*100:.0f}%\n"
        f"Severity Score: {severity}\n"
        f"RVI: {rvi}/100  |  Risk: {risk_level}\n"
        f"Cleanup: {cleanup}\n"
        f"Est. Personnel: {personnel:,}\n"
        f"- ODA Monitoring"
    )
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=body,
            from_=TWILIO_PHONE_NUMBER,
            to=phone
        )
        print(f"[sms] Sent to {phone}  SID={message.sid}")
    except Exception as exc:
        print(f"[sms] Failed to send to {phone} – {exc}")

# =============================================================================
# ── message.py functions (integrated inline) ─────────────────────────────────
# =============================================================================

GULF_MIN_LAT, GULF_MAX_LAT = 18.0, 31.0
GULF_MIN_LON, GULF_MAX_LON = -98.0, -80.0


def fetch_marine_species(lat: float, lon: float, radius_km: float = 100.0) -> list:
    import requests as _req
    if not (GULF_MIN_LAT <= lat <= GULF_MAX_LAT and GULF_MIN_LON <= lon <= GULF_MAX_LON):
        return []
    url    = "https://api.obis.org/v3/occurrence"
    params = {"decimalLatitude": lat, "decimalLongitude": lon,
              "radius": int(radius_km * 1000), "marine": True, "size": 3000}
    try:
        r = _req.get(url, params=params, timeout=5)  # short timeout – non-critical
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    relevant = {"Actinopterygii","Chondrichthyes","Anthozoa","Malacostraca",
                "Cephalopoda","Bivalvia","Mammalia","Reptilia"}
    out = []
    for rec in data.get("results", []):
        cls  = rec.get("class")
        sci  = rec.get("scientificName")
        bathy = rec.get("bathymetry")
        if cls in relevant and sci and bathy is not None and bathy < 0:
            out.append({"scientific": sci,
                        "common": rec.get("vernacularName", sci),
                        "class": cls})
    return out


def compute_relative_vulnerability(species_list: list,
                                   spill_area_sqkm: float,
                                   oil_type: str) -> tuple:
    if not species_list:
        return 0.0, "LOW"
    vuln_factors = {"Anthozoa":1.8,"Mammalia":1.7,"Reptilia":1.6,
                    "Chondrichthyes":1.4,"Actinopterygii":1.0,
                    "Malacostraca":0.9,"Cephalopoda":0.9,"Bivalvia":0.8}
    counts = defaultdict(int)
    for sp in species_list:
        counts[sp["class"]] += 1
    vuln_score = sum(n * vuln_factors.get(c, 1.0) for c, n in counts.items())
    norm_vuln  = min(vuln_score / 500.0, 1.0)
    area_factor = min(spill_area_sqkm / 100.0, 1.0)
    oil_mod = {"light": 0.8, "crude": 1.0, "heavy": 1.5}.get(oil_type.lower(), 1.0)
    rvi = (0.6 * norm_vuln + 0.4 * area_factor) * oil_mod * 100 / 1.5
    rvi = max(0.0, min(round(rvi, 2), 100.0))
    level = ("ECO-CRITICAL" if rvi >= 75 else
             "HIGH"         if rvi >= 50 else
             "MODERATE"     if rvi >= 25 else "LOW")
    return rvi, level


def _build_personnel_model() -> RandomForestRegressor:
    X = np.array([[200,0],[1000,1],[3000,2],[10000,2],[50000,2]])
    y = np.array([120, 450, 1400, 4200, 47000])
    m = RandomForestRegressor(n_estimators=20, random_state=42)
    m.fit(X, y)
    return m

_PERSONNEL_MODEL = _build_personnel_model()


def predict_personnel(volume_barrels: float, oil_type: str) -> int:
    code = {"light":0,"crude":1,"heavy":2}.get(oil_type.lower(), 1)
    return int(round(_PERSONNEL_MODEL.predict([[volume_barrels, code]])[0]))


def recommend_cleanup(oil_type: str, risk_level: str) -> str:
    if risk_level in ("HIGH", "ECO-CRITICAL"):
        return ("Mechanical Recovery + Dedicated Wildlife Response Teams "
                "(NOAA sensitive habitat protocol)")
    if oil_type.lower() == "heavy":
        return "Containment Booms + Limited/Authorized Dispersants (for persistent oils)"
    return "Skimmers + Sorbent Booms (standard mechanical recovery)"


def generate_alert_text(area_sqm: float, volume_barrels: float,
                        oil_type: str, rvi: float, risk_level: str,
                        personnel: int, cleanup: str,
                        lat: float = 28.5, lon: float = -89.5) -> str:
    area_sqkm = area_sqm / 1e6
    return f"""\
🚨 ODA MARINE ECOSYSTEM SPILL ALERT 🚨
Simulated Gulf of Mexico Incident
Location: {lat:.4f}°N, {lon:.4f}°W

Spill Characteristics:
  • Area: {area_sqkm:.4f} sq km  ({area_sqm:.2f} m²)
  • Estimated Volume: {volume_barrels:,.1f} barrels
  • Oil Type: {oil_type.capitalize()}

Ecological Assessment:
  • Relative Vulnerability Index (RVI): {rvi}/100
  • Risk Level: {risk_level}

Response Estimate:
  • Approx. Personnel Required: {personnel:,}
  • Recommended Cleanup Strategy: {cleanup}

Note: Personnel estimate is illustrative only.
Real-world example: Deepwater Horizon peak response ~47,000 people.

Scientific Sources:
  BOEM RESA • NOAA ESI • EPA Ecological Risk Guidelines (1998)
  Deepwater Horizon NRDA • OBIS marine biodiversity database
"""


def send_email_alert(subject: str, body: str,
                     recipient: str,
                     attachment_bytes: bytes = None,
                     attachment_name: str = "report.pdf"):
    """Send an email via Gmail SMTP. attachment_bytes is optional PDF bytes."""
    try:
        msg = MIMEMultipart()
        msg['From']    = SENDER_EMAIL
        msg['To']      = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        if attachment_bytes:
            part = MIMEApplication(attachment_bytes, _subtype='pdf')
            part.add_header('Content-Disposition', 'attachment',
                            filename=attachment_name)
            msg.attach(part)

        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(SENDER_EMAIL, SENDER_APP_PASS)
            s.send_message(msg)   # send_message handles binary safely
        print(f"[email] Sent to {recipient}")
    except Exception as e:
        print(f"[email] Failed – {e}")


# =============================================================================
# PDF generator
# =============================================================================

def _abs_img(path: str) -> str:
    """Convert a relative or URL static/ path to an absolute filesystem path."""
    if not path:
        return ''
    clean = path.lstrip('/')          # strip leading slash from URL paths (/static/...)
    if os.path.isabs(clean):
        return clean
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, clean)


class _OdaPDF(FPDF):
    """Custom FPDF subclass with header/footer."""
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 85, 170)
        self.cell(0, 10, 'ODA  -  Oil Spill Analysis Report', align='C')
        self.ln(4)
        self.set_draw_color(0, 85, 170)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(),
                  self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10,
                  'Sources: BOEM RESA | NOAA ESI | EPA Eco-Risk Guidelines (1998) | OBIS',
                  align='C')


def generate_pdf_report(username: str, area_sqm: float,
                        volume_m3: float, volume_gallons: float,
                        thickness_um: float, oil_type: str,
                        rvi: float, risk_level: str,
                        cleanup: str, personnel: int,
                        segmented_image_path: str = None,
                        overlay_image_path: str = None) -> bytes:
    """Build a valid PDF using fpdf2 and return raw bytes."""
    pdf = _OdaPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── helper lambdas ────────────────────────────────────────────────────
    def section(title: str):
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(0, 51, 136)
        pdf.set_fill_color(230, 240, 255)
        pdf.cell(0, 8, title, fill=True)
        pdf.ln(2)
        pdf.set_text_color(30, 30, 30)

    def row(label: str, value: str):
        pdf.set_x(pdf.l_margin)          # always start from the left margin
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(70, 7, label + ':')
        pdf.set_font('Helvetica', '', 10)
        val_w = pdf.w - pdf.r_margin - pdf.get_x()   # explicit remaining width
        pdf.multi_cell(val_w, 7, str(value), new_x='LMARGIN', new_y='NEXT')

    # ── User & date ───────────────────────────────────────────────────────
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(80, 80, 80)
    from datetime import datetime
    pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}   |   User: {username}')
    pdf.ln(8)

    # ── Spill Measurements ────────────────────────────────────────────────
    section('Spill Measurements')
    pdf.ln(2)
    row('Detected Area',   f'{area_sqm:,.2f} m2  ({area_sqm / 1e6:.6f} km2)')
    row('Oil Type',        oil_type.capitalize())
    row('Thickness',       f'{thickness_um} um')
    row('Volume',          f'{volume_m3:.6f} cubic metres  /  {volume_gallons:,.2f} gallons')
    pdf.ln(6)

    # ── Ecological Intelligence ───────────────────────────────────────────
    section('Ecological Intelligence (OBIS-based)')
    pdf.ln(2)
    row('RVI Score',       f'{rvi} / 100')
    row('Risk Level',      risk_level)
    row('Cleanup Strategy', cleanup)
    row('Est. Personnel',  f'{personnel:,}')
    pdf.ln(6)

    # ── Images ───────────────────────────────────────────────────────────
    abs_seg     = _abs_img(segmented_image_path)
    abs_overlay = _abs_img(overlay_image_path)
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    img_w  = page_w / 2 - 4   # two images side by side

    has_seg     = bool(abs_seg     and os.path.exists(abs_seg))
    has_overlay = bool(abs_overlay and os.path.exists(abs_overlay))

    if has_seg or has_overlay:
        section('Detection Images')
        pdf.ln(3)

    if has_seg and has_overlay:
        # Side by side
        x_start = pdf.l_margin
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(img_w + 4, 6, 'Segmentation Map', align='C')
        pdf.cell(img_w + 4, 6, 'Detection Overlay', align='C')
        pdf.ln(6)
        y = pdf.get_y()
        pdf.image(abs_seg,     x=x_start,             y=y, w=img_w)
        pdf.image(abs_overlay, x=x_start + img_w + 8, y=y, w=img_w)
        pdf.ln(img_w * 0.75 + 4)   # approximate height advance
    elif has_seg:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 6, 'Segmentation Map', align='C'); pdf.ln(6)
        pdf.image(abs_seg, x=pdf.l_margin, w=page_w * 0.8)
    elif has_overlay:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 6, 'Detection Overlay', align='C'); pdf.ln(6)
        pdf.image(abs_overlay, x=pdf.l_margin, w=page_w * 0.8)

    # ── Return bytes ──────────────────────────────────────────────────────
    return bytes(pdf.output())


# =============================================================================
# Model architecture
# =============================================================================

class HamidaEtAl(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, dilation=1):
        super().__init__()
        self.patch_size      = patch_size
        self.input_channels  = input_channels
        dilation             = (dilation, 1, 1)
        self.conv1 = nn.Conv3d(1,  20, (3,3,3), stride=(1,1,1), dilation=dilation, padding=1)
        self.pool1 = nn.Conv3d(20, 20, (3,1,1), dilation=dilation, stride=(2,1,1), padding=(1,0,0))
        self.conv2 = nn.Conv3d(20, 35, (3,3,3), dilation=dilation, stride=(1,1,1), padding=(1,0,0))
        self.pool2 = nn.Conv3d(35, 35, (3,1,1), dilation=dilation, stride=(2,1,1), padding=(1,0,0))
        self.conv3 = nn.Conv3d(35, 35, (3,1,1), dilation=dilation, stride=(1,1,1), padding=(1,0,0))
        self.conv4 = nn.Conv3d(35, 35, (2,1,1), dilation=dilation, stride=(2,1,1), padding=(1,0,0))
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1,1,self.input_channels,self.patch_size,self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x); x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        return self.fc(x)


# =============================================================================
# Segmentation helpers
# =============================================================================

def segment_full_image(model, image, patch_size, device, batch_size=512):
    channels, H, W = image.shape
    pad    = patch_size // 2
    padded = np.pad(image, ((0,0),(pad,pad),(pad,pad)), mode='reflect')

    all_patches = np.empty((H * W, channels, patch_size, patch_size), dtype=np.float32)
    idx = 0
    for i in range(H):
        for j in range(W):
            all_patches[idx] = padded[:, i:i+patch_size, j:j+patch_size]
            idx += 1

    tensor = torch.from_numpy(all_patches).unsqueeze(1)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            out  = model(tensor[start:start+batch_size].to(device))
            preds.append(torch.argmax(out, dim=1).cpu().numpy())
    return np.concatenate(preds).reshape(H, W).astype(np.int64)


def calculate_area(seg, pixel_width=3.3, pixel_height=3.3):
    return int(np.count_nonzero(seg)) * pixel_width * pixel_height


def run_segmentation_task(task_id: str, file_path: str,
                          user_email: str, username: str, oil_type: str):
    """Background worker – segmentation + eco-intelligence email."""
    try:
        tasks[task_id]['status'] = 'running'

        input_channels, n_classes, patch_size = 34, 2, 3
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = HamidaEtAl(input_channels, n_classes, patch_size).to(device)
        state_dict = torch.load(
            'E:/codewithsenpai/ODA/ODA_OIL_SPILL/ODA(OIL)/oil_spill_detection'
            '/oil_spill_detection/file.pth',
            map_location=device
        )
        model.load_state_dict(state_dict)

        mat = scipy.io.loadmat(file_path)
        if 'img' not in mat:
            tasks[task_id] = {'status': 'error',
                               'error': "Variable 'img' not found in .mat file."}
            return

        full_image = mat['img']
        H, W, C    = full_image.shape

        pca  = PCA(n_components=input_channels)
        data = pca.fit_transform(full_image.reshape(-1, C))
        img_reduced = data.reshape(H, W, input_channels).transpose(2, 0, 1)

        seg   = segment_full_image(model, img_reduced, patch_size, device)
        area  = calculate_area(seg)

        uploads_dir = 'static/uploads'
        seg_path     = os.path.join(uploads_dir, f'segmented_{task_id}.png')
        overlay_path = os.path.join(uploads_dir, f'overlay_{task_id}.png')

        plt.imsave(seg_path, seg, cmap='jet')

        rgb = img_reduced[:3].transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb); ax.imshow(seg, cmap='jet', alpha=0.4); ax.axis('off')
        fig.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Build URL-safe paths (leading /, forward slashes) for <img src>
        seg_url     = '/static/uploads/' + f'segmented_{task_id}.png'
        overlay_url = '/static/uploads/' + f'overlay_{task_id}.png'
        tasks[task_id] = {
            'status': 'done', 'area': area,
            'segmented_image': seg_url, 'overlay_image': overlay_url,
            'oil_type': oil_type, 'username': username,
            'user_email': user_email,
        }

        # Email is sent once only – from calculate_volume after the user
        # chooses thickness/units, so we have the full data to build the PDF.

    except Exception as exc:
        tasks[task_id] = {'status': 'error', 'error': str(exc)}


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def home():
    return render_template('base.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE username=%s AND password=%s",
                    (username, password)
                )
                user = cur.fetchone()
            if user:
                session['username'] = username
                session['email']    = user.get('email', '')
                session['phone']    = user.get('phone', '')
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials.', 'danger')
        except Exception as e:
            flash(f"Database error: {e}", "danger")
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email    = request.form.get('email', '').strip()
        # Normalize phone to E.164 before storing
        raw_phone = request.form.get('phone', '').strip()
        phone     = re.sub(r'[\s\-\(\)\.]', '', raw_phone)
        if phone and not phone.startswith('+'):
            phone = '+' + phone
        if phone and not re.fullmatch(r'\+[1-9]\d{6,14}', phone):
            flash('Phone must be in international format, e.g. +919959583328', 'danger')
            return render_template('signup.html')
        try:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE username=%s", (username,))
                if cur.fetchone():
                    flash('Username already exists.', 'danger')
                else:
                    cur.execute(
                        "INSERT INTO users (username, password, email, phone) VALUES(%s,%s,%s,%s)",
                        (username, password, email, phone)
                    )
                    conn.commit()
                    flash('Signup successful! You can now log in.', 'success')
                    return redirect(url_for('login'))
        except Exception as e:
            flash(f"Database error: {e}", "danger")
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please log in.', 'warning')
        return redirect(url_for('login'))
    activities = [
        {'date':'2024-03-20','method':'Hyperspectral','filename':'sample1.mat',
         'status':'Completed','status_color':'success'},
        {'date':'2024-03-19','method':'SAR','filename':'sample2.tiff',
         'status':'Processing','status_color':'warning'},
    ]
    return render_template('dashboard.html', activities=activities)


# ── Hyperspectral upload ──────────────────────────────────────────────────────

@app.route('/upload_hyperspectral', methods=['GET', 'POST'])
def upload_hyperspectral():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            uploads_dir = 'static/uploads'
            os.makedirs(uploads_dir, exist_ok=True)
            file_path = os.path.join(uploads_dir, file.filename)
            file.save(file_path)

            if file.filename.lower().endswith(('.jpg', '.jpeg')):
                try:
                    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
                    h, w, c = img.shape
                    exp = np.tile(img, (1, 1, 34 // c + 1))[:, :, :34]
                    mat_path = os.path.join(uploads_dir,
                                            os.path.splitext(file.filename)[0] + '.mat')
                    scipy.io.savemat(mat_path, {'img': exp})
                    flash('Converted to MAT. Upload the .mat file to proceed.', 'success')
                    return redirect(request.url)
                except Exception as e:
                    flash(f'Conversion error: {e}', 'danger')
                    return redirect(request.url)

            elif file.filename.lower().endswith('.mat'):
                oil_type = request.form.get('oil_type', 'crude')
                task_id  = str(uuid.uuid4())
                tasks[task_id] = {'status': 'pending'}
                threading.Thread(
                    target=run_segmentation_task,
                    args=(task_id, file_path,
                          session.get('email', ''),
                          session.get('username', 'anonymous'),
                          oil_type),
                    daemon=True
                ).start()
                return redirect(url_for('processing', task_id=task_id))
            else:
                flash('Upload a JPG (for conversion) or a .mat file.', 'warning')
                return redirect(request.url)

    return render_template('upload_hyperspectral.html')


# ── Processing / polling ──────────────────────────────────────────────────────

@app.route('/processing/<task_id>')
def processing(task_id):
    return render_template('processing.html', task_id=task_id)


@app.route('/task_status/<task_id>')
def task_status(task_id):
    task = tasks.get(task_id)
    if task is None:
        return jsonify({'status': 'not_found'}), 404
    if task['status'] == 'done':
        return jsonify({'status': 'done',
                        'redirect': url_for('show_result', task_id=task_id)})
    if task['status'] == 'error':
        return jsonify({'status': 'error', 'error': task.get('error', 'Unknown')})
    return jsonify({'status': task['status']})


@app.route('/result/<task_id>')
def show_result(task_id):
    task = tasks.get(task_id, {})
    if task.get('status') != 'done':
        flash('Result not ready.', 'warning')
        return redirect(url_for('upload_hyperspectral'))
    return render_template('volume_calculation.html',
                           area=task['area'],
                           segmented_image=task['segmented_image'],
                           overlay_image=task['overlay_image'],
                           oil_type=task.get('oil_type', 'crude'))


# ── Volume calculation with PDF + email ──────────────────────────────────────

@app.route('/calculate_volume', methods=['POST'])
def calculate_volume():
    area         = float(request.form['area'])
    unit_choice  = request.form['units']
    thickness_um = float(request.form.get('thickness', 1))
    oil_type     = request.form.get('oil_type', 'crude')
    seg_img      = request.form.get('segmented_image', '')
    overlay_img  = request.form.get('overlay_image', '')

    volume_m3      = area * (thickness_um * 1e-6)
    gallons_per_m3 = 264.17
    volume_gallons = volume_m3 * gallons_per_m3
    volume_barrels = volume_m3 / 0.158987

    if unit_choice == 'cubic_meters':
        volume_display = f"Estimated Volume: {volume_m3:.6f} cubic metres  (based on {thickness_um} um thickness)"
    elif unit_choice == 'gallons':
        volume_display = f"Estimated Volume: {volume_gallons:,.2f} gallons  (based on {thickness_um} um thickness)"
    else:
        volume_display = (f"Estimated Volume: {volume_m3:.6f} cubic metres  /  "
                          f"{volume_gallons:,.2f} gallons  (based on {thickness_um} um thickness)")

    user_email = session.get('email', '')
    username   = session.get('username', 'anonymous')

    # ── Build eco-intelligence data (quick, no API call on critical path) ─
    area_sqkm  = area / 1e6
    species    = fetch_marine_species(28.5, -89.5)
    rvi, level = compute_relative_vulnerability(species, area_sqkm, oil_type)
    personnel  = predict_personnel(volume_barrels, oil_type)
    cleanup    = recommend_cleanup(oil_type, level)

    # ── Generate PDF and save to disk so the user can download it ─────────
    outputs_dir = os.path.join('static', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    report_name = f"ODA_Report_{uuid.uuid4().hex[:8]}.pdf"
    report_path = os.path.join(outputs_dir, report_name)

    pdf_bytes = generate_pdf_report(
        username=username,
        area_sqm=area,
        volume_m3=volume_m3,
        volume_gallons=volume_gallons,
        thickness_um=thickness_um,
        oil_type=oil_type,
        rvi=rvi, risk_level=level,
        cleanup=cleanup, personnel=personnel,
        segmented_image_path=seg_img,
        overlay_image_path=overlay_img,
    )
    with open(report_path, 'wb') as f:
        f.write(pdf_bytes)

    # ── Send email + SMS in background (one send only) ───────────────────
    if user_email or session.get('phone'):
        body = generate_alert_text(area, volume_barrels, oil_type,
                                   rvi, level, personnel, cleanup)
        user_phone = session.get('phone', '')
        def _send():
            if user_email:
                send_email_alert(
                    subject="ODA - Full Spill Analysis Report",
                    body=body,
                    recipient=user_email,
                    attachment_bytes=pdf_bytes,
                    attachment_name=report_name,
                )
            if user_phone:
                send_sms_alert(user_phone, area, oil_type,
                               rvi, level, personnel, cleanup)
        threading.Thread(target=_send, daemon=True).start()

    return render_template('results.html',
                           volume_display=volume_display,
                           area=area,
                           volume_m3=volume_m3,
                           volume_gallons=volume_gallons,
                           rvi=rvi, risk_level=level,
                           cleanup=cleanup, personnel=personnel,
                           report_path=report_path)


@app.route('/download_report')
def download_report():
    path = request.args.get('path', '')
    abs_path = _abs_img(path) if path else ''
    if not abs_path or not os.path.exists(abs_path):
        flash('Report not found.', 'danger')
        return redirect(url_for('dashboard'))
    return send_file(abs_path, as_attachment=True,
                     download_name='ODA_Spill_Report.pdf',
                     mimetype='application/pdf')


# ── SAR upload ────────────────────────────────────────────────────────────────

@app.route('/upload_sar', methods=['GET', 'POST'])
def upload_sar():
    if request.method == 'GET':
        return render_template('upload_sar.html')

    file = request.files.get('file')
    if not file or file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(request.url)

    uploads_dir = os.path.join('static', 'uploads')
    outputs_dir = os.path.join('static', 'outputs')
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    try:
        rf         = Roboflow(api_key="2YQXWnYr8GzfpiemnPRm")
        project    = rf.workspace().project("oil-spill-yolo")
        rf_model   = project.version(1).model
        prediction = rf_model.predict(file_path)
        out_path   = os.path.join(outputs_dir, f"predicted_{file.filename}")
        prediction.save(out_path)

        # Compute oil area from Roboflow bounding-box predictions
        # Assumes 10 m SAR pixel spacing (Sentinel-1 IW default)
        SAR_PIXEL_M = 10.0
        oil_area    = 0.0
        try:
            preds_json = prediction.json()
            # Roboflow may return a dict with 'predictions' OR a raw list
            if isinstance(preds_json, dict):
                boxes = preds_json.get('predictions', [])
            elif isinstance(preds_json, list):
                boxes = preds_json
            else:
                boxes = []
            print(f"[SAR] Roboflow JSON sample: {str(preds_json)[:300]}")
            if boxes:
                oil_area = sum(
                    p.get('width', p.get('w', 0)) * p.get('height', p.get('h', 0))
                    for p in boxes
                ) * (SAR_PIXEL_M ** 2)
        except Exception as _je:
            print(f"[SAR] prediction.json() failed: {_je}")

        if oil_area <= 0:
            # Fallback: dark-pixel mask on the ORIGINAL SAR image
            orig_img = cv2.imread(file_path)
            gray     = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
            blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
            mask     = cv2.adaptiveThreshold(blurred, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
            kernel   = np.ones((3, 3), np.uint8)
            mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
            mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            oil_area = int(np.sum(mask == 255)) * (SAR_PIXEL_M ** 2)
            print(f"[SAR] Used mask fallback. oil_area={oil_area}")

        orig = Image.open(file_path)
        seg  = Image.open(out_path)
        if orig.size != seg.size:
            seg = seg.resize(orig.size)
        side = Image.new('RGB', (orig.width * 2, orig.height))
        side.paste(orig, (0, 0)); side.paste(seg, (orig.width, 0))
        sbs_path = os.path.join(outputs_dir, f"side_by_side_{file.filename}")
        side.save(sbs_path)

        # ── Email + SMS alert for SAR analysis ───────────────────────────
        user_email = session.get('email', '')
        user_phone = session.get('phone', '')
        username   = session.get('username', 'anonymous')
        oil_type   = request.form.get('oil_type', 'crude')
        if user_email or user_phone:
            def _sar_email():
                area_sqkm  = oil_area / 1e6
                vol_est    = area_sqkm * 6.28
                species    = fetch_marine_species(28.5, -89.5)
                rvi, level = compute_relative_vulnerability(species, area_sqkm, oil_type)
                personnel  = predict_personnel(vol_est, oil_type)
                cleanup    = recommend_cleanup(oil_type, level)
                body       = generate_alert_text(oil_area, vol_est, oil_type,
                                                 rvi, level, personnel, cleanup)
                if user_email:
                    send_email_alert(
                        subject="ODA - SAR Spill Detection Alert",
                        body=body,
                        recipient=user_email
                    )
                if user_phone:
                    send_sms_alert(user_phone, oil_area, oil_type,
                                   rvi, level, personnel, cleanup)
            threading.Thread(target=_sar_email, daemon=True).start()

        # Use absolute URL paths so <img src> works from any route
        sbs_url = '/' + sbs_path.replace('\\', '/')
        out_url = '/' + out_path.replace('\\', '/')
        return render_template('volume_calculation.html',
                               area=oil_area,
                               segmented_image=sbs_url,
                               overlay_image=out_url,
                               oil_type=oil_type)

    except Exception as e:
        flash(f'Error processing image: {e}', 'danger')
        return redirect(request.url)


# =============================================================================
if __name__ == '__main__':
    app.run(debug=False, port=5002)
