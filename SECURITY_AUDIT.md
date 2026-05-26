# 🔒 รายงานตรวจสอบความปลอดภัยโปรเจกต์ Doc-AI2

**วันที่ตรวจสอบ:** 2026-05-26
**ผู้ตรวจสอบ:** Security Audit System
**ขอบเขต:** ไฟล์ทั้งหมดในโปรเจกต์ (src/*.py, config, Dockerfile, requirements.txt, docs)

---

## 📊 สรุปผลการตรวจสอบ

| ระดับความรุนแรง | จำนวนช่องโหว่ | สถานะ |
|:---:|:---:|:---|
| 🔴 **วิกฤต (Critical)** | 3 | ต้องแก้ไขทันที |
| 🟠 **สูง (High)** | 5 | แก้ไขก่อนขึ้น GitHub |
| 🟡 **ปานกลาง (Medium)** | 7 | ควรแก้ไข |
| 🟢 **ต่ำ (Low)** | 4 | แนะนำให้ปรับปรุง |

---

## 🔴 ช่องโหว่ระดับวิกฤต (Critical)

### CVE-001: ไม่มีไฟล์ .gitignore — ข้อมูลอ่อนไหวอาจถูก commit ขึ้น GitHub

**ไฟล์ที่เกี่ยวข้อง:** ไม่มี `.gitignore` ในโปรเจกต์
**ระดับความรุนแรง:** 🔴 วิกฤต

**ปัญหาที่พบ:**
- โปรเจกต์ไม่มีไฟล์ `.gitignore` เลย
- ไฟล์ `unsloth_dataset.json` (ข้อมูล training dataset) อาจมีข้อมูลอ่อนไหว
- ไดเรกทอรี `data/`, `cache/`, `output/` อาจมีไฟล์ PDF เอกสารลับ
- ไฟล์ `.env` ที่มี credentials จะถูก commit ขึ้น GitHub ทันที

**วิธีแก้ไข:** สร้างไฟล์ `.gitignore` ที่ครอบคลุม (ดูส่วน "คำแนะนำ .gitignore" ด้านล่าง)

---

### CVE-002: Hardcoded Internal IP Addresses ในโค้ด

**ไฟล์ที่เกี่ยวข้อง:** [`config.py`](config.py:177), [`vector_db.py`](vector_db.py:56)
**ระดับความรุนแรง:** 🔴 วิกฤต

**ปัญหาที่พบ:**
```python
# config.py บรรทัด 177 — Hardcoded internal IP สำหรับ LLM server
"server_url": os.getenv("LLM_SERVER_URL", "http://10.10.1.212:8700"),

# vector_db.py บรรทัด 56 — Hardcoded internal IP สำหรับ embedding API
EXTERNAL_EMBEDDING_API_URL = os.getenv(
    "EXTERNAL_EMBEDDING_API_URL", "http://192.168.1.212:7700/v1"
)
```

**ความเสี่ยง:**
- การเปิดเผยโครงสร้างเครือข่ายภายใน (Internal Network Reconnaissance)
- ผู้โจมตีสามารถระบุ topology ของเครือข่ายได้
- IP address อาจถูกใช้สำหรับการโจมตีแบบ targeted

**โค้ดก่อนแก้ไข:**
```python
# config.py
llm_config = {
    "server_url": os.getenv("LLM_SERVER_URL", "http://10.10.1.212:8700"),
}

# vector_db.py
EXTERNAL_EMBEDDING_API_URL = os.getenv(
    "EXTERNAL_EMBEDDING_API_URL", "http://192.168.1.212:7700/v1"
)
```

**โค้ดหลังแก้ไข:**
```python
# config.py
llm_config = {
    "server_url": os.getenv("LLM_SERVER_URL", "http://localhost:8080"),
}

# vector_db.py
EXTERNAL_EMBEDDING_API_URL = os.getenv(
    "EXTERNAL_EMBEDDING_API_URL", "http://localhost:7700/v1"
)
```

---

### CVE-003: Default Password ใน openwebui_server.py

**ไฟล์ที่เกี่ยวข้อง:** [`openwebui_server.py`](openwebui_server.py:47)
**ระดับความรุนแรง:** 🔴 วิกฤต

**ปัญหาที่พบ:**
```python
# openwebui_server.py บรรทัด 47
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
```
ค่า default เป็น `"password"` ซึ่งเป็นการเปิดเผยว่า username/password เริ่มต้นของระบบคืออะไร

**โค้ดก่อนแก้ไข:**
```python
class AppConfig:
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")  # ⚠️ DANGEROUS
```

**โค้ดหลังแก้ไข:**
```python
class AppConfig:
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD") or _require_env("NEO4J_PASSWORD")

def _require_env(key: str) -> str:
    """Raise error if critical environment variable is missing."""
    raise ValueError(
        f"Missing required environment variable: {key}\n"
        f"Set it via .env file or export before running."
    )
```

---

## 🟠 ช่องโหว่ระดับสูง (High)

### CVE-004: การใช้ HTTP แทน HTTPS ในการเชื่อมต่อทั้งหมด

**ไฟล์ที่เกี่ยวข้อง:** [`config.py`](config.py:177), [`vector_db.py`](vector_db.py:56), [`llm_extractor.py`](llm_extractor.py:59), [`microservice_v2.py`](microservice_v2.py:407)
**ระดับความรุนแรง:** 🟠 สูง

**ปัญหาที่พบ:**
- LLM Server ใช้ `http://` แทน `https://`
- Embedding API ใช้ `http://`
- Microservice เปิดรับเชื่อมต่อผ่าน HTTP ล้วน
- Qdrant client เชื่อมต่อแบบไม่เข้ารหัส

**ความเสี่ยง:**
- ข้อมูลถูกส่งแบบ plain text (Man-in-the-Middle attack)
- API keys และ tokens อาจถูกดักจับได้
- ข้อมูลเอกสาร PDF ที่แปลงเป็น Markdown อาจรั่วไหล

**โค้ดก่อนแก้ไข:**
```python
# llm_extractor.py บรรทัด 59
LLM_SERVER_URL: str = os.getenv("LLM_SERVER_URL", "http://localhost:8080")

# microservice_v2.py บรรทัด 407
logger.info(f"Microservice started on http://{MICROSERVICE_HOST}:{MICROSERVICE_PORT}")
```

**โค้ดหลังแก้ไข:**
```python
# llm_extractor.py — เพิ่มการตรวจสอบ protocol
LLM_SERVER_URL: str = os.getenv("LLM_SERVER_URL", "http://localhost:8080")
if not LLM_SERVER_URL.startswith(("http://localhost", "http://127.0.0.1", "https://")):
    if not LLM_SERVER_URL.startswith("https://"):
        logger.warning("LLM_SERVER_URL uses HTTP in production. Consider HTTPS.")

# microservice_v2.py — เพิ่มตัวเลือก TLS/SSL
ENABLE_TLS = os.getenv('ENABLE_TLS', 'false').lower() == 'true'
TLS_CERT = os.getenv('TLS_CERT_PATH', None)
TLS_KEY = os.getenv('TLS_KEY_PATH', None)
```

---

### CVE-005: ไม่มีการ Authentication/Authorization สำหรับ API Endpoints

**ไฟล์ที่เกี่ยวข้อง:** [`microservice_v2.py`](microservice_v2.py), [`openwebui_server.py`](openwebui_server.py)
**ระดับความรุนแรง:** 🟠 สูง

**ปัญหาที่พบ:**
- `/convert` endpoint ใน microservice ไม่มีการตรวจสอบสิทธิ์
- `/api/search` endpoint ใน OpenWebUI server เปิดให้ทุกคนเข้าถึง
- CORS ตั้งค่าเป็น `allow_origins=["*"]` (อนุญาตทุก domain)
- ไม่มี API key authentication หรือ JWT token validation

**โค้ดก่อนแก้ไข:**
```python
# openwebui_server.py บรรทัด 109-115
app.add_middleware(
    CORSMiddleware,
    allow_origins=AppConfig.CORS_ORIGINS,  # default: ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**โค้ดหลังแก้ไข:**
```python
# openwebui_server.py — เพิ่ม API Key Authentication
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(API_KEY_HEADER)) -> str:
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        logger.warning("API_KEY not set — skipping authentication")
        return api_key
    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# จำกัด CORS สำหรับ production
CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["http://localhost:3000"]
```

---

### CVE-006: Hardcoded API Key Placeholder ใน marker_run.py

**ไฟล์ที่เกี่ยวข้อง:** [`marker_run.py`](marker_run.py:42)
**ระดับความรุนแรง:** 🟠 สูง

**ปัญหาที่พบ:**
```python
# marker_run.py บรรทัด 42
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "sk-placeholder")
```
แม้จะเป็น placeholder แต่รูปแบบ `sk-` อาจทำให้เครื่องมือตรวจสอบ security คิดว่าเป็น real key ที่รั่วไหล

**โค้ดก่อนแก้ไข:**
```python
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "sk-placeholder")
```

**โค้ดหลังแก้ไข:**
```python
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
if not DEFAULT_API_KEY and os.getenv("REQUIRE_OPENAI_KEY", "false").lower() == "true":
    raise ValueError("OPENAI_API_KEY environment variable is required")
```

---

### CVE-007: การเชื่อมต่อ Redis โดยไม่เข้ารหัส

**ไฟล์ที่เกี่ยวข้อง:** [`rate_limiter.py`](rate_limiter.py:83)
**ระดับความรุนแรง:** 🟠 สูง

**ปัญหาที่พบ:**
```python
# rate_limiter.py บรรทัด 83
self.redis = redis_lib.Redis(
    host=host,
    port=port,
    password=password,
    db=db,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)
```
ไม่มีการรองรับ Redis over TLS/SSL

**โค้ดหลังแก้ไข:**
```python
self.redis = redis_lib.Redis(
    host=host,
    port=port,
    password=password,
    db=db,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    ssl=os.getenv('REDIS_SSL', 'false').lower() == 'true',
    ssl_cert_reqs='required' if os.getenv('REDIS_SSL') else None,
)
```

---

### CVE-008: Neo4j Connection ใช้ neo4j:// แทน neo4j+s:// (encrypted)

**ไฟล์ที่เกี่ยวข้อง:** [`config.py`](config.py:250)
**ระดับความรุนแรง:** 🟠 สูง

**ปัญหาที่พบ:**
```python
'neo4j_uri': os.getenv('NEO4J_URI', 'neo4j://10.10.1.210:7687'),
```
ใช้ `neo4j://` (ไม่เข้ารหัส) แทน `neo4j+s://` (encrypted) หรือ `neo4j+ssc://` (with cert verification)

**โค้ดหลังแก้ไข:**
```python
'neo4j_uri': os.getenv('NEO4J_URI', 'neo4j://localhost:7687'),
# Note: For production, use neo4j+s:// (encrypted) or neo4j+ssc:// (with cert verification)
```

---

## 🟡 ช่องโหว่ระดับปานกลาง (Medium)

### CVE-009: Dependencies ล้าสมัยและไม่มี version pin ที่ชัดเจน

**ไฟล์ที่เกี่ยวข้อง:** [`requirements.txt`](requirements.txt)
**ระดับความรุนแรง:** 🟡 ปานกลาง

**ปัญหาที่พบ:**
```
torch>=2.0.0           # ควรใช้เวอร์ชันเฉพาะเพื่อ reproducibility
openai>=0.27.0         # ⚠️ เวอร์ชัน 0.x ล้าสมัยมาก — openai v1.x มี breaking changes
neo4j>=5.0.0           # ควรตรวจสอบ CVE ล่าสุด
requests>=2.25.0       # ⚠️ requests <2.31.0 มี CVE-2023-32681
qdrant-client>=1.0.0   # ควรใช้เวอร์ชันเฉพาะ
```

**ความเสี่ยง:**
- `openai>=0.27.0`: เวอร์ชัน 0.x ไม่รองรับ API ใหม่ อาจมีช่องโหว่ด้านความปลอดภัย
- `requests>=2.25.0`: มี CVE ที่ทราบแล้วในเวอร์ชันเก่า
- ไม่มี hash checking สำหรับ dependencies

**วิธีแก้ไข:**
```txt
# requirements.txt — ใช้เวอร์ชันที่ตรวจสอบแล้ว
torch>=2.0.0,<3.0.0
openai>=1.0.0,<2.0.0          # อัปเดตเป็น v1.x
requests>=2.31.0,<3.0.0        # แก้ CVE-2023-32681
neo4j>=5.17.0,<6.0.0
qdrant-client>=1.7.0,<2.0.0
sentence-transformers>=2.2.0,<3.0.0
aiohttp>=3.9.0,<4.0.0
uvicorn>=0.27.0,<1.0.0
```

---

### CVE-010: Log อาจบันทึกข้อมูลอ่อนไหว

**ไฟล์ที่เกี่ยวข้อง:** [`openwebui_server.py`](openwebui_server.py:154), [`microservice_v2.py`](microservice_v2.py:348)
**ระดับความรุนแรง:** 🟡 ปานกลาง

**ปัญหาที่พบ:**
```python
# openwebui_server.py บรรทัด 154-157
logger.info(
    "Search request: query=%r, top_k=%d, max_hops=%d",
    request.search_query, top_k, max_hops,
)
```
การ log search query อาจบันทึกข้อมูลอ่อนไหวจากผู้ใช้

**โค้ดหลังแก้ไข:**
```python
# เพิ่มการ mask ข้อมูลใน log
import re

def sanitize_for_log(text: str, max_length: int = 50) -> str:
    """Sanitize text for logging — truncate and mask sensitive patterns."""
    if len(text) > max_length:
        text = text[:max_length] + "..."
    # Mask potential API keys or tokens in logs
    text = re.sub(r'(sk-)\w+', r'\1****', text)
    text = re.sub(r'([A-Za-z0-9+/]{40,})', '****', text)
    return text

logger.info(
    "Search request: query=%r, top_k=%d, max_hops=%d",
    sanitize_for_log(request.search_query), top_k, max_hops,
)
```

---

### CVE-011: subprocess.call โดยไม่ตรวจสอบ input ที่ปลอดภัย

**ไฟล์ที่เกี่ยวข้อง:** [`main.py`](main.py:153)
**ระดับความรุนแรง:** 🟡 ปานกลาง

**ปัญหาที่พบ:**
```python
# main.py บรรทัด 153-157
subprocess.run(['soffice', '--headless', '--convert-to', 'pdf',
                pub_path, '--outdir', out_dir],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
```
แม้จะใช้ list form (ปลอดภัยจาก shell injection) แต่ `pub_path` และ `out_dir` ควรตรวจสอบว่าเป็น path ที่ถูกต้อง

**โค้ดหลังแก้ไข:**
```python
# เพิ่มการตรวจสอบ path ก่อนส่งไปยัง subprocess
def _safe_subprocess_convert(pub_path: str, out_dir: str):
    # ตรวจสอบว่าไฟล์มีอยู่จริงและเป็นไฟล์ (ไม่ใช่ directory)
    if not os.path.isfile(pub_path):
        raise ValueError(f"Input path is not a file: {pub_path}")
    # ตรวจสอบชื่อไฟล์ไม่ให้มีตัวอักษรอันตราย
    basename = os.path.basename(pub_path)
    if '..' in basename or ';' in basename or '&' in basename:
        raise ValueError(f"Unsafe filename detected: {basename}")
    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf',
                    pub_path, '--outdir', out_dir],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
```

---

### CVE-012: CORS เปิดกว้างเกินไป (Wildcard Origins)

**ไฟล์ที่เกี่ยวข้อง:** [`openwebui_server.py`](openwebui_server.py:52)
**ระดับความรุนแรง:** 🟡 ปานกลาง

**ปัญหาที่พบ:**
```python
CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
```
ค่า default คือ `*` ซึ่งอนุญาตให้ทุก domain เรียก API ได้

**โค้ดหลังแก้ไข:**
```python
# ใช้ empty list เป็น default — ต้องตั้งค่าเองใน production
CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS: list = [o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()] if CORS_ORIGINS_RAW else []
if not CORS_ORIGINS:
    logger.warning("CORS_ORIGINS not set — no external origins allowed by default")
```

---

### CVE-013: Dockerfile ใช้ root user ก่อนสลับไป appuser

**ไฟล์ที่เกี่ยวข้อง:** [`Dockerfile`](Dockerfile)
**ระดับความรุนแรง:** 🟡 ปานกลาง

**ปัญหาที่พบ:**
- Stage 2 ทำงานเป็น root ระหว่างติดตั้ง package และ copy ไฟล์
- แม้จะสลับไป `appuser` ในตอนท้าย แต่ระหว่าง build ยังมีความเสี่ยง

**โค้ดหลังแก้ไข:**
```dockerfile
# เพิ่ม security headers และลด attack surface
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove

# Copy only necessary files (not .env, data/, cache/)
COPY --chown=appuser:appuser openwebui_tool.py .
COPY --chown=appuser:appuser openwebui_server.py .
# ... etc
```

---

### CVE-014: ไม่มีการเข้ารหัสข้อมูลใน Vector Database

**ไฟล์ที่เกี่ยวข้อง:** [`vector_db.py`](vector_db.py), [`neo4j_vector_store.py`](neo4j_vector_store.py)
**ระดับความรุนแรง:** 🟡 ปานกลาง

**ปัญหาที่พบ:**
- Qdrant เก็บ embedding และ metadata โดยไม่เข้ารหัส
- Neo4j เก็บข้อมูลเอกสารโดยไม่เข้ารหัส
- ไม่มี encryption at rest สำหรับข้อมูลใน database

**คำแนะนำ:**
- เปิดใช้งาน TLS สำหรับ Qdrant (`qdrant://` แทน `http://`)
- ใช้ Neo4j Aurora หรือเปิด enable encryption ใน self-managed
- พิจารณาใช้ encrypted volume สำหรับ data directory

---

### CVE-015: Rate Limiting ใช้ IP address เป็น identifier

**ไฟล์ที่เกี่ยวข้อง:** [`microservice_v2.py`](microservice_v2.py:201)
**ระดับความรุนแรง:** 🟡 ปานกลาง

**ปัญหาที่พบ:**
```python
remote = request.remote or "unknown"
allowed, info = limiter.is_allowed(remote)
```
การใช้ IP address เพียงอย่างเดียวอาจถูก bypass ผ่าน proxy หรือ CDN

**โค้ดหลังแก้ไข:**
```python
# ใช้ combination ของ IP + User-Agent สำหรับ rate limiting ที่แม่นยำกว่า
def _get_client_identifier(request: web.Request) -> str:
    ip = request.remote or "unknown"
    # ตรวจสอบ X-Forwarded-For สำหรับ proxy/CDN
    forwarded = request.headers.get("X-Forwarded-For", ip)
    user_agent = request.headers.get("User-Agent", "unknown")[:50]
    return f"{forwarded}:{user_agent}"

client_id = _get_client_identifier(request)
allowed, info = limiter.is_allowed(client_id)
```

---

### CVE-016: File Upload ไม่ตรวจสอบ MIME Type อย่างเข้มงวด

**ไฟล์ที่เกี่ยวข้อง:** [`microservice_v2.py`](microservice_v2.py:307)
**ระดับความรุนแรง:** 🟡 ปานกลาง

**ปัญหาที่พบ:**
ตรวจสอบเฉพาะ magic bytes `%PDF` แต่ไม่ตรวจสอบ Content-Type header หรือ file extension ที่ส่งมา

**โค้ดหลังแก้ไข:**
```python
# ตรวจสอบหลายชั้น
filename = pdf_file.filename or ""
if not filename.lower().endswith('.pdf'):
    raise ValueError("File must have .pdf extension")

# ตรวจสอบ magic bytes
with open(temp_pdf_path, 'rb') as f:
    magic_bytes = f.read(5)
    if not magic_bytes.startswith(b'%PDF'):
        raise ValueError("Uploaded file is not a valid PDF (invalid magic bytes)")

# ตรวจสอบ Content-Disposition header
content_type = pdf_file.content_type or ""
if content_type and content_type != "application/pdf":
    logger.warning(f"Content-Type mismatch: expected application/pdf, got {content_type}")
```

---

## 🟢 ช่องโหว่ระดับต่ำ (Low)

### CVE-017: Debug/Info Logging เปิดอยู่ใน production

**ไฟล์ที่เกี่ยวข้อง:** หลายไฟล์
**ระดับความรุนแรง:** 🟢 ต่ำ

**คำแนะนำ:** เพิ่ม environment variable สำหรับควบคุม log level:
```python
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
```

---

### CVE-018: ไม่มีการจำกัดขนาดของ request body ใน OpenWebUI server

**ไฟล์ที่เกี่ยวข้อง:** [`openwebui_server.py`](openwebui_server.py)
**ระดับความรุนแรง:** 🟢 ต่ำ

**คำแนะนำ:** เพิ่ม max_request_size ใน FastAPI:
```python
app = FastAPI(
    # ...
)
# หรือจำกัดใน Pydantic model
class SearchRequest(BaseModel):
    search_query: str = Field(..., min_length=1, max_length=2000)  # ✅ มีแล้ว
```

---

### CVE-019: unsloth_dataset.json อาจมีข้อมูลอ่อนไหว

**ไฟล์ที่เกี่ยวข้อง:** [`unsloth_dataset.json`](unsloth_dataset.json)
**ระดับความรุนแรง:** 🟢 ต่ำ

**ปัญหาที่พบ:** ไฟล์ JSON dataset ถูกเก็บใน repository โดยตรง ซึ่งอาจมีเนื้อหาเอกสารที่เป็นความลับ

**คำแนะนำ:** เพิ่มใน `.gitignore` หรือใช้ LFS สำหรับไฟล์ขนาดใหญ่

---

### CVE-020: Prompt Injection ใน LLM extraction

**ไฟล์ที่เกี่ยวข้อง:** [`llm_extractor.py`](llm_extractor.py)
**ระดับความรุนแรง:** 🟢 ต่ำ

**ปัญหาที่พบ:** เนื้อหาเอกสารถูกส่งไปยัง LLM โดยตรงโดยไม่มีการกรอง ซึ่งอาจทำให้ผู้โจมตี inject prompt ผ่านเนื้อหา PDF

**คำแนะนำ:** เพิ่มการ sanitize input ก่อนส่งไปยัง LLM:
```python
def sanitize_llm_input(text: str) -> str:
    """Remove potential prompt injection patterns from document text."""
    # ลบคำสั่งที่คล้าย system prompt
    injection_patterns = [
        r'(?i)ignore\s+previous\s+instructions',
        r'(?i)system\s*:.*',
        r'(?i)you\s+are\s+now\s+',
        r'(?i)forget\s+everything',
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, '[REDACTED]', text, flags=re.IGNORECASE)
    return text
```

---

## 📋 คำแนะนำ .gitignore ที่สมบูรณ์

```gitignore
# ============================================
# Python
# ============================================
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
.eggs/
*.egg

# ============================================
# Virtual Environments
# ============================================
.venv/
.venv2/
venv/
env/
ENV/

# ============================================
# IDE / Editor
# ============================================
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# ============================================
# Environment & Secrets
# ============================================
.env
.env.local
.env.*.local
!.env.example
*.key
*.pem
*.crt
*.p12
*.pfx
*.jks
*.keystore

# ============================================
# Data & Cache
# ============================================
data/
cache/
*.joblib
*.pkl
*.pt
*.pth
*.bin
*.onnx

# ============================================
# Output & Generated Files
# ============================================
output/
*.md.backup
unsloth_dataset.json
*.jsonl

# ============================================
# Logs
# ============================================
*.log
logs/

# ============================================
# Testing
# ============================================
.pytest_cache/
.coverage
htmlcov/
.tox/

# ============================================
# Docker
# ============================================
.dockerignore.backup

# ============================================
# OS
# ============================================
Thumbs.db
desktop.ini

# ============================================
# Large files (use Git LFS instead)
# ============================================
*.pdf
*.pub
*.xlsx
*.csv
*.parquet
*.gguf
```

---

## 🔐 คำแนะนำ .env.example

```bash
# ============================================
# Neo4j Graph Database
# ============================================
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<CHANGE_ME_STRONG_PASSWORD>
NEO4J_DATABASE=neo4j

# ============================================
# Qdrant Vector Database
# ============================================
QDRANT_URI=localhost:6333
QDRANT_API_KEY=<CHANGE_ME_API_KEY>
QDRANT_COLLECTION=markdown_embeddings

# ============================================
# LLM Server (Ollama / vLLM / llama.cpp)
# ============================================
LLM_SERVER_URL=http://localhost:8080
LLM_MODEL=Qwen3.6-27B-MTP-GGUF
LLM_TEMPERATURE=0.0
LLM_ENABLE_THINKING=false
LLM_MAX_TOKENS=4096
LLM_TIMEOUT=120

# ============================================
# Embedding Model
# ============================================
EMBEDDING_MODEL=all-MiniLM-L6-v2
EXTERNAL_EMBEDDING_API_URL=http://localhost:7700/v1

# ============================================
# OpenAI API (สำหรับ Marker LLM mode)
# ============================================
OPENAI_API_KEY=<CHANGE_ME_OPENAI_KEY>

# ============================================
# Redis (Rate Limiting)
# ============================================
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=<OPTIONAL_REDIS_PASSWORD>
REDIS_DB=0
REDIS_SSL=false

# ============================================
# Microservice
# ============================================
MICROSERVICE_HOST=localhost
MICROSERVICE_PORT=8080
MAX_UPLOAD_SIZE_MB=100
ENABLE_TLS=false
TLS_CERT_PATH=
TLS_KEY_PATH=

# ============================================
# API Security
# ============================================
API_KEY=<CHANGE_ME_API_KEY>
CORS_ORIGINS=http://localhost:3000

# ============================================
# Logging & Monitoring
# ============================================
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090

# ============================================
# Processing
# ============================================
PARALLEL_PROCESSING=true
PARALLEL_MAX_WORKERS=0
VRAM_SEQUENTIAL_MODE=true
VRAM_SAFE_LIMIT_GB=22
```

---

## 🛡️ คำแนะนำการตั้งค่า GitHub Security

### 1. เปิดใช้งาน GitHub Secret Scanning

ไปที่ **Repository Settings → Code security → Secret scanning** แล้วเปิดใช้งาน:
- ✅ Secret scanning
- ✅ Push protection (ป้องกัน push ที่มี secrets)

### 2. ตั้งค่า Pre-commit Hooks

ติดตั้ง pre-commit hooks เพื่อตรวจสอบ secrets ก่อน commit:

```bash
pip install pre-commit trufflehog gitleaks
```

สร้างไฟล์ `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: detect-private-key
      - id: detect-aws-credentials
      - id: check-added-large-files
        args: ['--maxkb=1024']

  - repo: https://github.com/zricethezav/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  - repo: https://github.com/trufflesecurity/trufflehog
    rev: v3.63.1
    hooks:
      - id: trufflehog
```

### 3. ตั้งค่า CodeQL Analysis

สร้างไฟล์ `.github/workflows/codeql.yml`:

```yaml
name: "CodeQL Security Analysis"
on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  analyze:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    strategy:
      matrix:
        language: ['python']
    steps:
      - uses: actions/checkout@v4
      - uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
      - uses: github/codeql-action/analyze@v3
```

---

## 📝 สรุปขั้นตอนการแก้ไขที่ต้องทำก่อนขึ้น GitHub

| ลำดับ | การกระทำ | ความสำคัญ |
|:---:|:---|:---:|
| 1 | สร้างไฟล์ `.gitignore` ตามคำแนะนำด้านบน | 🔴 วิกฤต |
| 2 | ลบหรือย้าย `unsloth_dataset.json` ออกจาก repository | 🔴 วิกฤต |
| 3 | เปลี่ยน hardcoded IP เป็น localhost หรือ env variable | 🔴 วิกฤต |
| 4 | ลบ default password `"password"` จาก openwebui_server.py | 🔴 วิกฤต |
| 5 | สร้างไฟล์ `.env.example` และเพิ่มใน repository | 🟠 สูง |
| 6 | อัปเดต `requirements.txt` เป็นเวอร์ชันที่ปลอดภัย | 🟠 สูง |
| 7 | เพิ่ม API Key Authentication สำหรับ endpoints | 🟠 สูง |
| 8 | ตั้งค่า pre-commit hooks สำหรับตรวจสอบ secrets | 🟠 สูง |
| 9 | เปิดใช้งาน GitHub Secret Scanning | 🟡 ปานกลาง |
| 10 | เพิ่มการ sanitize log output | 🟡 ปานกลาง |
| 11 | พิจารณาใช้ HTTPS/TLS สำหรับการเชื่อมต่อทั้งหมด | 🟡 ปานกลาง |
| 12 | ตรวจสอบ git history ว่ามี secrets ที่เคย commit ไปแล้วหรือไม่ | 🔴 วิกฤต |

### การลบ Secrets ออกจาก Git History

หากเคย commit ไฟล์ที่มี secrets ไปแล้ว ให้ใช้:

```bash
# ติดตั้ง git-filter-repo
pip install git-filter-repo

# ลบไฟล์ .env ออกจาก history ทั้งหมด
git filter-repo --force --path .env --invert-paths

# ลบไฟล์ dataset ออกจาก history
git filter-repo --force --path unsloth_dataset.json --invert-paths

# Push ใหม่ (force)
git push --force
```

---

**หมายเหตุ:** รายงานนี้ครอบคลุมการตรวจสอบโค้ดหลัก ไฟล์คอนฟิก Dockerfile และเอกสารประกอบ ควรทำการตรวจสอบซ้ำก่อน每一次 deploy ไปยัง production environment
