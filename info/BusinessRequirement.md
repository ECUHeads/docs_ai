

# 🚀 ใบงาน: Universal GraphRAG & Cognitive Content Engine

**Tech Stack Overview:** * **OS:** Ubuntu Server 24.04

* **Hardware:** 1x NVIDIA RTX 3090 (24GB VRAM) *[ต้องจัดการ VRAM แบบ Sequential]*
* **Core LLM:** Qwen 3.6 27B (MTP-GGUF) รันผ่าน `llama-server`
* **Document Parsing:** Marker
* **Graph/Vector Database:** Neo4j
* **Frontend/RAG Interface:** Open WebUI (ใช้ Built-in Tools / Pipelines แทนระบบภายนอก)
* **Audio TTS:** Omnivoice

---

## 📌 1. GraphDB (Neo4j) Meta-Schema Design

การออกแบบ Schema นี้ต้องเป็นแบบครอบจักรวาล (Universal) รองรับทั้งสาย Quant, วิศวกรรม, งานคอนเทนต์ และงานเขียนนิยาย

### 1.1 Node Labels (Entities หลัก)

Dev ต้องเขียน Cypher เพื่อสกัดข้อมูลลง Label เหล่านี้เป็นหลัก:

* `KnowledgeConcept`: ทฤษฎี, สมการคณิตศาสตร์, แนวคิดการเงิน
* `Hardware`: ชิ้นส่วน IC, สเปคเซิร์ฟเวอร์
* `Strategy`: กลยุทธ์การเทรด, Alpha
* `Narrative`: องค์ประกอบนิยาย (ตัวละคร, สถานที่, เหตุการณ์)
* `ContentAsset`: โครงสร้างหนังสือ (บท, หัวข้อ) หรือโครงสร้างคลิปวิดีโอ

### 1.2 Node Properties (Dynamic Metadata)

ทุก Node ไม่ว่าจะ Label ใด **ต้องมี** Properties พื้นฐานเหล่านี้เพื่อรองรับ Multilingual และ Content Creation:

* `id`: (String) Unique ID
* `domain`: (String) หมวดหมู่หลัก (เช่น finance, hardware, fiction)
* `name_th` / `name_en`: (String) ชื่อคอนเซปต์
* `technical_desc_th` / `technical_desc_en`: (String) คำอธิบายเชิงลึก
* `layman_explanation`: (String) คำอธิบายภาษาคนทั่วไป (ใช้ทำสคริปต์)
* `analogy`: (String) การเปรียบเทียบกับของใกล้ตัว
* `visual_concept`: (String) Prompt สำหรับทำภาพประกอบ
* `fiction_seed`: (String) ไอเดียพล็อตนิยายจากความรู้นี้
* `embedding`: (List[Float]) Vector Embedding ของเนื้อหานี้ (สำหรับ Vector Search)

### 1.3 Relationships (Edges)

แบ่งตามมิติการใช้งาน:

* **Logical (ความจริง):** `PREDICTS`, `APPLIES_TO`, `CORRELATES_WITH`, `REQUIRES_INFRA`
* **Creative (คอนเทนต์/จินตนาการ):** `INSPIRES_PLOT`, `IS_EXAMPLE_OF`, `CONTRADICTS`
* **Structural (หนังสือ):** `HAS_CHAPTER`, `NEXT_CHAPTER`, `REFERENCES_CONCEPT`

---

## 📌 2. Ingestion Pipeline (การสกัดข้อมูลลง Graph & Vector)

Dev ต้องเขียน Python Script (Orchestrator) ทำงานเป็นลำดับขั้นดังนี้:

1. **Parsing & Preprocessing:**
* ใช้ **Marker** สกัด PDF เป็น Markdown
* ฝัง YAML Frontmatter ระบุ `domain` และ `language` ที่หัวไฟล์
* ใช้ `MarkdownHeaderTextSplitter` หั่นไฟล์เป็น Chunk


2. **LLM Extraction (ผ่าน Qwen 3.6):**
* ยิง API ไปที่ `llama-server` (เปิดใช้ `--spec-type draft-mtp`)
* **Strict Rule:** ต้องบังคับพารามิเตอร์ `temperature=0.0` และ `enable_thinking=False` เพื่อป้องกัน JSON พัง
* System Prompt ต้องบังคับให้ออกมาเป็นโครงสร้าง Meta-Schema (มี layman, analogy, fiction_seed ครบถ้วน)


3. **Dual-Storage Routing (เก็บข้อมูล 2 ขา):**
* **ขา Graph:** นำ JSON แปลงเป็น Cypher `MERGE` query เพื่อสร้าง Node และ Relationship ลง Neo4j
* **ขา Vector:** นำข้อความ Chunk ดิบไปทำ Embedding (ใช้โมเดล Embedding ขนาดเล็ก เช่น `bge-m3` รันบน CPU หรือแชร์ VRAM) และอัปเดตลง Property `embedding` ในโหนดนั้นๆ ของ Neo4j (Neo4j รองรับ Vector Index)



---

## 📌 3. Hybrid Search Process (กลไกการดึงข้อมูล)

เมื่อต้องการดึงข้อมูลไปทำสคริปต์หรือตอบคำถาม Dev ต้องเขียนฟังก์ชัน `retrieve_context(query)` ที่ผสมผสาน 2 ทักษะ:

1. **Step 1: Vector Search (Semantic Match)**
* แปลงคำถามของผู้ใช้เป็น Vector
* Query เข้า Neo4j Vector Index ค้นหา Top-3 Nodes ที่ความหมายใกล้เคียงที่สุด


2. **Step 2: Graph Traversal (Logical Expansion)**
* ใช้ Cypher Query กระจายวง (Hop) ออกจาก Top-3 Nodes นั้นไปอีก 1-2 ชั้น
* *ตัวอย่าง Cypher:* `MATCH (n)-[r]->(neighbor) WHERE n.id IN $vector_results RETURN n, r, neighbor`


3. **Step 3: Context Assembly (การแพ็คข้อมูล)**
* นำข้อมูลที่ได้มาจัดฟอร์แมตใหม่ ดึง `layman_explanation`, `analogy`, และ `fiction_seed` มารวมกันเป็น Markdown Context Block เพื่อเตรียมส่งให้ LLM



---

## 📌 4. Open WebUI Integration (ตั้งค่า GraphRAG ให้ LLM เรียกใช้)

เพื่อให้ Qwen 3.6 สามารถคุยและดึงข้อมูลจาก GraphDB ได้ผ่านหน้าต่างแชทของ Open WebUI โดยตรง ให้ดำเนินการดังนี้:

1. **สร้าง Custom Tool (Function Calling) ใน Open WebUI:**
* ไปที่เมนู Workspace -> Tools ใน Open WebUI
* เขียน Python Code สร้าง Tool ชื่อ `search_knowledge_graph`


2. **โค้ดจำลองสำหรับ Tool ใน Open WebUI:**
```python
from neo4j import GraphDatabase
# ภายใน class ของ Tool
def search_knowledge_graph(self, search_query: str) -> str:
    """
    Use this tool to search the Universal Knowledge Graph for concepts, equations, hardware, and creative seeds.
    """
    # 1. รันกระบวนการ Hybrid Search (เรียก API หรือ Query ตรงไปที่ Neo4j)
    # 2. จัดฟอร์แมตผลลัพธ์
    # 3. Return เป็น String กลับไปให้ LLM อ่าน
    return formatted_hybrid_context

```


3. **การทำงาน (Execution Flow):**
* ผู้ใช้พิมพ์คำสั่งใน Open WebUI: *"ช่วยคิดพล็อตนิยายจากสถาปัตยกรรม Low-latency C++ หน่อย"*
* Qwen 3.6 (ที่ต่อผ่าน OpenAI API Format เข้า Open WebUI) จะตัดสินใจเรียกใช้ Tool `search_knowledge_graph`
* ระบบวิ่งไปดึงข้อมูลจาก Neo4j (ได้ technical_desc, fiction_seed)
* Qwen 3.6 นำข้อมูลนั้นมาแต่งเป็นพล็อตนิยายและพิมพ์ตอบกลับในหน้าแชท



---

### ⚠️ หมายเหตุสำคัญสำหรับ Developer (VRAM Management)

เนื่องจากระบบใช้ RTX 3090 ใบเดียว (VRAM 24GB) **ห้ามรันการสร้างเสียง (Omnivoice), การสร้างภาพ (ComfyUI), และ LLM (Qwen 3.6) พร้อมกันเด็ดขาด** ในสคริปต์ทำ Automated Content (Phase ผลิตสื่อ) ต้องเขียน Logic แบบ **Sequential Load/Unload**:

1. โหลด Qwen 3.6 -> Gen สคริปต์/ดึง Graph -> Unload โมเดล
2. โหลด Omnivoice -> Gen เสียง -> Unload โมเดล
3. โหลด ComfyUI -> Gen ภาพประกอบ -> Unload โมเดล