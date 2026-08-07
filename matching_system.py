import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from datetime import datetime
import pandas as pd
import numpy as np
import re
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
from gensim.utils import simple_preprocess
from sentence_transformers import SentenceTransformer
# Optional parsers for PDF/DOCX
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx
except Exception:
    docx = None

# Optional: language name normalization (no hardcoded language list)
try:
    import pycountry
except Exception:
    pycountry = None

# 1. Charger le dataset des offres d'emploi
df_jobs = pd.read_csv('postings.csv')

# --- Lightweight caches (module-level) ---
_JOB_ARTIFACTS_CACHE = {}
_MODEL_CACHE = {}


def _jobs_signature(jobs_df):
    """Builds a stable-ish signature for caching artifacts."""
    try:
        src_mtime = os.path.getmtime('postings.csv')
    except Exception:
        src_mtime = None
    cols = tuple(sorted([str(c) for c in jobs_df.columns]))
    return (int(len(jobs_df)), cols, src_mtime)


def _get_sentence_model(model_name='all-MiniLM-L6-v2'):
    model = _MODEL_CACHE.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
    return model


def _get_job_text_series(jobs_df, desc_col=None):
    if 'job_text' in jobs_df.columns:
        return jobs_df['job_text']
    return jobs_df.apply(lambda r: build_job_text(r, desc_col=desc_col), axis=1)


def _get_job_artifacts(jobs_df, desc_col=None, tfidf_max_features=5000):
    """Returns cached job artifacts used for matching."""
    sig = _jobs_signature(jobs_df)
    cached = _JOB_ARTIFACTS_CACHE.get(sig)
    if cached is not None:
        return cached

    job_texts_raw = _get_job_text_series(jobs_df, desc_col=desc_col).fillna('').astype(str)
    job_texts_clean = job_texts_raw.apply(preprocess)
    job_titles = jobs_df.get('job_title', pd.Series([''] * len(jobs_df))).fillna('').astype(str)

    # TF-IDF artifacts (fit on jobs once; CV is transformed later)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=int(tfidf_max_features))
    job_tfidf_matrix = vectorizer.fit_transform(job_texts_clean.tolist())

    # LDA artifacts (trained once on jobs for stability + speed)
    lda_num_topics = 5
    try:
        tokenized_jobs = [simple_preprocess(t) for t in job_texts_clean.tolist()]
        lda_dictionary = corpora.Dictionary(tokenized_jobs)
        if len(tokenized_jobs) >= 10:
            lda_dictionary.filter_extremes(no_below=2, no_above=0.6)
        lda_corpus = [lda_dictionary.doc2bow(t) for t in tokenized_jobs]

        if len(lda_dictionary) == 0:
            raise ValueError("Empty LDA dictionary")

        lda_model = models.LdaModel(
            lda_corpus,
            num_topics=int(lda_num_topics),
            id2word=lda_dictionary,
            passes=2,
            random_state=42,
            minimum_probability=0.0,
        )

        job_topic_vectors = []
        for bow in lda_corpus:
            topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
            vec = np.array([prob for _, prob in topics], dtype=float)
            if vec.size != lda_num_topics:
                padded = np.zeros(lda_num_topics, dtype=float)
                padded[:min(lda_num_topics, vec.size)] = vec[:min(lda_num_topics, vec.size)]
                vec = padded
            job_topic_vectors.append(vec)
        job_topic_vectors = (
            np.vstack(job_topic_vectors)
            if job_topic_vectors
            else np.zeros((len(jobs_df), lda_num_topics), dtype=float)
        )
    except Exception as e:
        print(f"LDA artifacts build failed: {e}")
        lda_dictionary = None
        lda_model = None
        job_topic_vectors = np.zeros((len(jobs_df), lda_num_topics), dtype=float)

    artifacts = {
        'signature': sig,
        'job_texts_raw': job_texts_raw.tolist(),
        'job_texts_clean': job_texts_clean.tolist(),
        'job_titles': job_titles.tolist(),
        'tfidf_vectorizer': vectorizer,
        'job_tfidf_matrix': job_tfidf_matrix,
        'lda_num_topics': int(lda_num_topics),
        'lda_dictionary': lda_dictionary,
        'lda_model': lda_model,
        'job_topic_vectors': job_topic_vectors,
        # Parsed jobs are built lazily per index
        'parsed_jobs': {},
    }
    _JOB_ARTIFACTS_CACHE[sig] = artifacts
    return artifacts


def _get_parsed_job(artifacts, idx):
    pj = artifacts['parsed_jobs'].get(idx)
    if pj is None:
        pj = parse_job_posting(artifacts['job_texts_raw'][idx])
        artifacts['parsed_jobs'][idx] = pj
    return pj

# 2. Simuler des CVs (format texte)
cvs = [
    {
        'name': 'Alice',
        'text': "Data engineer with 3 years experience in Python, Spark, ETL, AWS, and big data pipelines."
    },
    {
        'name': 'Bob',
        'text': "Experienced in SQL, Airflow, cloud data platforms, and machine learning for data engineering."
    },
    {
        'name': 'Charlie',
        'text': "Junior data engineer, strong in Scala, Hadoop, and data warehousing."
    }
]

# 3. Prétraitement des textes
def preprocess(text):
    """Light normalization for NLP vectorizers.

    Keeps digits and common tech punctuation (+, #, ., -) so tokens like
    'c++', 'c#', 'python3', 'spark-streaming' are not destroyed.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    # Keep alphanumerics + a few common separators for tech tokens
    text = re.sub(r"[^a-z0-9\s\+\#\-\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_job_text(row, desc_col=None):
    """Builds a canonical job text used for all matchers.

    We intentionally include both summary/description and skill lists.
    """
    parts = []
    if desc_col and desc_col in row:
        parts.append(str(row.get(desc_col, "")))
    parts.append(str(row.get('job_summary', "")))
    parts.append(str(row.get('job_skills', "")))
    parts.append(str(row.get('job_title', "")))
    # Include level hints when available (helps seniority parsing)
    parts.append(str(row.get('job level', row.get('job_level', ''))))
    return " ".join([p for p in parts if p and p != 'nan'])

# Détecter la colonne contenant le texte de l'offre
possible_desc_cols = ['description', 'job_summary', 'job_description', 'summary', 'job_description_text']
job_text_col = next((c for c in possible_desc_cols if c in df_jobs.columns), None)
if job_text_col is None:
    # We'll still be able to match using job_summary/job_skills/job_title if present.
    job_text_col = None

# Nettoyer une version canonique du texte job (utilisée partout)
df_jobs['job_text'] = df_jobs.apply(lambda r: build_job_text(r, desc_col=job_text_col), axis=1)
df_jobs['description_clean'] = df_jobs['job_text'].fillna('').astype(str).apply(preprocess)
for cv in cvs:
    cv['text_clean'] = preprocess(cv['text'])

# Charger des CVs depuis un dossier 'cvs' (PDF/DOCX/TXT) s'il existe
def extract_text_from_pdf(path):
    if pdfplumber is None:
        raise ImportError("pdfplumber is required to parse PDFs. Install with `pip install pdfplumber`")
    text = ''
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text += txt + '\n'
    return text

def extract_text_from_docx(path):
    if docx is None:
        raise ImportError("python-docx is required to parse DOCX files. Install with `pip install python-docx`")
    d = docx.Document(path)
    return '\n'.join([p.text for p in d.paragraphs])

# --- Advanced Data Engineering Domain Knowledge & Normalization ---

# CEFR Language Levels Mapping
CEFR_LEVELS = {
    # Native-like
    'native': 6,
    'native speaker': 6,
    'mother tongue': 6,
    'bilingual': 6,

    # C2
    'c2': 5,
    'fluent': 5,
    'full professional': 5,
    'native-like': 5,

    # C1
    'c1': 4,
    'advanced': 4,
    'professional': 4,
    'professional working': 4,
    'highly proficient': 4,

    # B2
    'b2': 3,
    'upper intermediate': 3,
    'working proficiency': 3,
    'proficient': 3,

    # B1
    'b1': 2,
    'intermediate': 2,
    'conversational': 2,
    'working knowledge': 2,

    # A2
    'a2': 1,
    'elementary': 1,
    'basic': 1,

    # A1
    'a1': 0,
    'beginner': 0,
    None: 0
}

CEFR_REVERSE = {v: k for k, v in CEFR_LEVELS.items() if k and len(k) == 2}
CEFR_REVERSE[6] = 'native'

SKILL_GROUPS = {
    'programming': ['python', 'scala', 'java', 'go', 'rust', 'c++', 'javascript', 'typescript'],
    'sql': ['sql', 't-sql', 'pl/sql', 'postgresql', 'mysql', 'oracle', 'sql server'],
    'big_data': ['spark', 'hadoop', 'hive', 'flink', 'kafka', 'mapreduce', 'presto', 'trino'],
    'orchestration': ['airflow', 'dagster', 'prefect', 'luigi', 'nifi', 'control-m'],
    'cloud': ['aws', 'gcp', 'azure', 's3', 'ec2', 'lambda', 'glue', 'bigquery', 'dataproc', 'dataflow', 'redshift', 'synapse'],
    'warehouse': ['snowflake', 'redshift', 'bigquery', 'synapse', 'databricks', 'teradata', 'exadata'],
    'streaming': ['kafka', 'flink', 'kinesis', 'pub/sub', 'spark streaming', 'beam'],
    'modeling': ['dimensional modeling', 'dbt', 'star schema', 'snowflake schema', 'data vault'],
    'devops': ['docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'gitlab ci', 'github actions', 'ci/cd'],
    'tools': ['git', 'linux', 'bash', 'jira', 'confluence', 'splunk', 'grafana']
}

ALIAS_MAP = {
    'google cloud': 'gcp', 'google cloud platform': 'gcp',
    'amazon web services': 'aws', 'microsoft azure': 'azure',
    'big query': 'bigquery', 'data bricks': 'databricks', 'snow flake': 'snowflake',
    'postgresql': 'postgres', 't-sql': 'sql', 'pl/sql': 'sql',
    'apache spark': 'spark', 'apache kafka': 'kafka', 'apache airflow': 'airflow', 'apache hadoop': 'hadoop',
    'etl': 'pipelines', 'data pipeline': 'pipelines', 'data pipelines': 'pipelines',
    # Common platform/product aliases
    'dbt core': 'dbt', 'dbt-core': 'dbt',
    'amazon msk': 'kafka', 'aws msk': 'kafka', 'msk': 'kafka',
    'amazon emr': 'aws', 'emr': 'aws'
}

# --- Soft skills (structured, weighted, explainable) ---
SOFT_SKILLS = [
    "communication", "teamwork", "collaboration", "stakeholder management",
    "leadership", "mentoring", "ownership", "initiative", "problem solving",
    "critical thinking", "time management", "adaptability", "documentation"
]

SOFT_ALIASES = {
    "stakeholder": "stakeholder management",
    "presentations": "communication",
    "cross-functional": "collaboration",
    "ownership mindset": "ownership",
    "team player": "teamwork",
    "problem-solving": "problem solving",
    "problem solving skills": "problem solving",
    "strong communicator": "communication",
}

def parse_language_level(text):
    text = text.lower()
    for level, val in CEFR_LEVELS.items():
        if level and level in text:
            return val
    return 2 # Default to Intermediate if mentioned but no level

def infer_languages(text):
    def _norm_lang_name(name: str) -> str:
        name = (name or '').strip().lower()
        name = re.sub(r"[^a-z\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _display_lang_name(name: str) -> str:
        n = _norm_lang_name(name)
        if not n:
            return ''

        # Prefer pycountry canonical naming if present
        if pycountry is not None:
            try:
                # Try exact name
                lang_obj = pycountry.languages.get(name=n)
                if lang_obj and getattr(lang_obj, 'name', None):
                    return str(lang_obj.name)
            except Exception:
                pass
            try:
                # Try fuzzy lookup
                hits = pycountry.languages.search_fuzzy(n)
                if hits:
                    return str(getattr(hits[0], 'name', n))
            except Exception:
                pass

        # Fallback: title-case
        return " ".join([w.capitalize() for w in n.split()])

    languages = []
    seen = set()

    raw = '' if text is None else (text if isinstance(text, str) else str(text))
    raw_lines = [ln.strip() for ln in raw.splitlines()]

    # 1) Parse explicit "Language: Level" style patterns (line-local)
    # Example: "English: C1", "French - C2"
    pattern = re.compile(
        r"^\s*([a-z][a-z\s]{1,30}?)\s*(?:[:\-–])\s*([a-z0-9\s]{1,25})\s*$",
        flags=re.I,
    )
    for ln in raw_lines:
        m = pattern.match(ln)
        if not m:
            continue
        lang_raw = m.group(1)
        lvl_raw = m.group(2)
        lvl = parse_language_level(lvl_raw)
        disp = _display_lang_name(lang_raw)
        key = _norm_lang_name(disp)
        if not disp or key in seen:
            continue
        # Avoid obvious false-positives like "years: 5-7"
        if any(bad in key for bad in ['year', 'years', 'month', 'months', 'experience', 'skills', 'salary', 'languages']):
            continue
        seen.add(key)
        languages.append({'language': disp, 'level': lvl, 'confidence': 0.9})

    # 2) Parse a "Languages" section if present (CVs)
    in_lang_section = False
    section_stop = re.compile(r"^(experience|skills|education|projects|certifications|contact|profile|summary)\b", re.I)
    split_items = re.compile(r"[;,•\t\|]+")
    for ln in raw_lines:
        low = ln.lower().strip()
        if not low:
            continue

        if re.match(r"^languages?\b", low):
            in_lang_section = True
            continue

        if in_lang_section and section_stop.search(low):
            break

        if not in_lang_section:
            continue

        # Split line into items like: "English: C2, French: C1"
        for item in [x.strip() for x in split_items.split(ln) if x.strip()]:
            # Handle separators inside item
            item_clean = item.replace('–', '-').replace('—', '-').strip()

            # Case: "Arabic Native" (no punctuation)
            m3 = re.match(
                r"^([a-zA-Z\s]{2,30}?)\s+((?:native|fluent|professional|advanced|intermediate|beginner|[abc][12]))\b\s*$",
                item_clean,
                flags=re.I,
            )
            if m3:
                lang_raw = m3.group(1)
                lvl_raw = m3.group(2)
            else:
                m2 = re.match(r"^([a-zA-Z\s]{2,30})\s*(?:[:\-–])?\s*(.*)$", item_clean)
                if not m2:
                    continue
                lang_raw = m2.group(1)
                lvl_raw = (m2.group(2) or '').strip()

            disp = _display_lang_name(lang_raw)
            key = _norm_lang_name(disp)
            if not disp or key in seen:
                continue
            if any(bad in key for bad in ['year', 'years', 'month', 'months', 'experience', 'skills', 'salary']):
                continue
            lvl = parse_language_level(lvl_raw) if lvl_raw else 2
            conf = 0.75 if lvl_raw else 0.55
            seen.add(key)
            languages.append({'language': disp, 'level': lvl, 'confidence': conf})

    return languages

def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    for alias, canonical in ALIAS_MAP.items():
        text = re.sub(rf'\b{re.escape(alias)}\b', canonical, text)
    return text


def extract_soft_skills(text):
    t = normalize_text(text)
    t = t.replace("-", " ")
    found = set()
    for k, v in SOFT_ALIASES.items():
        t = t.replace(k, v)
    for s in SOFT_SKILLS:
        if re.search(rf"\b{re.escape(s)}\b", t):
            found.add(s)
    return sorted(found)

def extract_years(text):
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', text, re.I)
    if matches:
        return max(map(float, matches))
    return 0.0


def extract_required_years_from_job(text):
    """Extracts required years of experience from a job posting.

    Handles common patterns:
    - "5-7 years", "5 – 7 Years", "5 to 7 years"
    - "3+ years", "3 years+", "3 years plus"
    - "at least 5 years", "minimum 5 years"

    Returns the minimum years when a range is found.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    # Normalize unicode dashes to '-'
    t = t.replace('–', '-').replace('—', '-').replace('−', '-')

    # Range: 5-7 years / 5 to 7 years
    m = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b', t)
    if m:
        try:
            lo = float(m.group(1))
            hi = float(m.group(2))
            return float(min(lo, hi))
        except Exception:
            pass

    # Range where dash got stripped: "5 7 years"
    m = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b', t)
    if m:
        try:
            lo = float(m.group(1))
            hi = float(m.group(2))
            # Guard against accidental huge pairs (e.g., years like 2023 2024)
            if lo <= 60 and hi <= 60:
                return float(min(lo, hi))
        except Exception:
            pass

    # Minimum/at least
    m = re.search(r'\b(?:at\s+least|minimum|min\.?|>=|more\s+than)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    # "years of experience: 5" (and similar)
    m = re.search(r'\byears?\s+of\s+(?:relevant\s+)?(?:experience|exp)\b\s*[:\-]?\s*(\d+(?:\.\d+)?)', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    # 3+ years / 3 years+ / 3 years plus
    m = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)\b', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*\+\b', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:plus|\+)', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    # Fallback: keep your previous simple extraction
    m = re.search(r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    return 0.0


def infer_work_mode(text, fallback_job_type=None):
    """Infers work mode from job text or a job_type column.

    Returns one of: 'Remote', 'Hybrid', 'Onsite', or '' when unknown.
    """
    candidates = []
    if isinstance(fallback_job_type, str) and fallback_job_type.strip():
        candidates.append(fallback_job_type)
    if isinstance(text, str) and text.strip():
        candidates.append(text)
    t = " ".join(candidates).lower()
    if not t:
        return ''

    # Normalize
    t = t.replace('on-site', 'onsite').replace('on site', 'onsite')
    t = t.replace('work from home', 'remote').replace('wfh', 'remote').replace('telecommute', 'remote')
    t = t.replace('in office', 'onsite').replace('in-office', 'onsite').replace('inoffice', 'onsite')

    # Priority: hybrid > remote > onsite (hybrid implies both)
    if 'hybrid' in t:
        return 'Hybrid'
    if 'remote' in t:
        return 'Remote'
    if 'onsite' in t:
        return 'Onsite'
    return ''


def extract_work_preference(text):
    """Extracts a candidate's work-mode preference from CV text.

    Returns one of: 'Remote', 'Hybrid', 'Onsite', or None.
    If not found, returns None so it won't affect scoring.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    t = t.replace('on-site', 'onsite').replace('on site', 'onsite')
    t = t.replace('work from home', 'remote').replace('wfh', 'remote')
    # Common phrasing
    if re.search(r'\b(prefer|preferred|seeking|looking for|open to)\b[^\n]{0,40}\bhybrid\b', t):
        return 'Hybrid'
    if re.search(r'\b(prefer|preferred|seeking|looking for|open to)\b[^\n]{0,40}\bremote\b', t):
        return 'Remote'
    if re.search(r'\b(prefer|preferred|seeking|looking for|open to)\b[^\n]{0,40}\bonsite\b', t):
        return 'Onsite'
    # Fallback: explicit single keyword (lower confidence)
    if 'hybrid' in t:
        return 'Hybrid'
    if 'remote' in t:
        return 'Remote'
    if 'onsite' in t:
        return 'Onsite'
    return None

def calculate_duration_months(start_str, end_str=None):
    """Calculates months between two date strings (e.g., 'Jan 2020', '2022-12', 'Present')."""
    def parse_date(date_str):
        if not date_str: return None
        date_str = date_str.lower().strip().strip('-').strip()
        if any(k in date_str for k in ['present', 'today', 'now', 'current', 'till date']):
            return datetime.now()
        
        # Try common patterns
        # Handle '12/2020' or '12-2020'
        date_str = re.sub(r'[\s/.-]', ' ', date_str) # Normalize separator
        
        patterns = [
            '%b %Y', '%B %Y', '%m %Y', '%Y %m', '%Y'
        ]
        for p in patterns:
            try:
                return datetime.strptime(date_str, p)
            except ValueError:
                continue
        
        # Fallback for solo years
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
        if year_match:
            return datetime(int(year_match.group(1)), 1, 1)
            
        return None

    start = parse_date(start_str)
    end = parse_date(end_str) if end_str else datetime.now()
    
    if start and end:
        # Avoid negative duration if start > end (edge case in parsing)
        if start > end: start, end = end, start
        return (end.year - start.year) * 12 + (end.month - start.month)
    return 0

def extract_explicit_durations(text):
    """Detects strings like '3 years 4 months' and returns total months."""
    months = 0
    # Pattern for "X years" or "X yrs"
    years_match = re.findall(r'(\d+)\s*(?:years?|yrs?)', text, re.I)
    for y in years_match:
        months += int(float(y)) * 12
    # Pattern for "X months" or "X mos"
    m_match = re.findall(r'(\d+)\s*(?:months?|mos?|mths?)', text, re.I)
    for m in m_match:
        months += int(float(m))
    return months

def extract_experience_from_dates(text):
    """Detects date ranges and returns total months."""
    # Improved pattern to handle (2023 - 2024), 2020-Present, Jan 2022 to Dec 2022
    # Allows optional parenthesis/brackets and various separators
    date_range_pattern = r'(?:\(|\[)?((?:[a-z]{3,9}|[\d]{1,2})?[\s/.-]?\d{4})\s*(?:-|to|–|—|until|till)\s*([a-z0-9\s/.-]+)(?:\)|\])?'
    matches = re.findall(date_range_pattern, text, re.I)
    
    date_based_months = 0
    for start, end in matches:
        date_based_months += calculate_duration_months(start, end)
    
    explicit_months = extract_explicit_durations(text)
    return max(date_based_months, explicit_months)

def find_all_date_ranges(text):
    """Finds all date ranges in the text and returns list of (start, end) tuples."""
    # Pattern matches: "2023 - 2024", "(Jan 2020 - Present)", "March 2019 to Dec 2021"
    date_range_pattern = r'(?:\(|\[)?((?:[a-z]{3,9}|[\d]{1,2})?[\s/.-]?\d{4})\s*(?:-|to|–|—|until|till)\s*([a-z0-9\s/.-]+)(?:\)|\])?'
    matches = re.findall(date_range_pattern, text, re.I)
    return matches

def format_duration(months):
    years = months // 12
    remaining_months = months % 12
    parts = []
    if years > 0: parts.append(f"{years} year{'s' if years > 1 else ''}")
    if remaining_months > 0: parts.append(f"{remaining_months} month{'s' if remaining_months > 1 else ''}")
    return ", ".join(parts) if parts else "0 months"

def detect_seniority(text, years=0):
    text = text.lower()
    if any(k in text for k in ['lead', 'principal', 'staff', 'architect']): return 'Lead'
    if any(k in text for k in ['senior', 'sr', 'expert']) or years >= 6: return 'Senior'
    if any(k in text for k in ['junior', 'jr', 'entry', 'intern']) or (0 < years < 2): return 'Junior'
    return 'Mid'

def parse_structured_cv(text):
    text_norm = normalize_text(text)
    fields = {
        'total_years': extract_years(text),
        'seniority': 'Mid',
        'skills_modeled': {},  # {skill: {level: X, sources: []}}
        'languages': infer_languages(text),
        'soft_skills': extract_soft_skills(text),
        'work_preference': extract_work_preference(text),
        'certifications': [],
        'education': '',
        'hobbies': [],
        'expected_salary': None,
        'experience': [], # list of {role, company, duration, context, tools}
        'projects': []    # list of {name, context, tools, description}
    }
    
    # 1. Split into sections - improved to handle UPPERCASE headers and multi-word headers
    sections = re.split(r'\n(?=[A-Z\s]*(?:EXPERIENCE|SKILLS|EDUCATION|PROJECTS|HOBBIES|INTERESTS|LANGUAGES|SUMMARY|CONTACT|PROFILE|WORK|CERTIF|DIPLOMA))', text, flags=re.I)
    
    current_section = "summary"
    all_experience_text = ""
    
    # FIRST: Scan entire CV for all date ranges and calculate total months
    all_date_ranges = find_all_date_ranges(text)
    total_months_from_dates = sum([calculate_duration_months(start, end) for start, end in all_date_ranges])
    
    # ALSO: Scan entire text for skills (fallback if section split fails)
    text_norm_full = normalize_text(text)
    for group, keywords in SKILL_GROUPS.items():
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', text_norm_full):
                if kw not in fields['skills_modeled']:
                    fields['skills_modeled'][kw] = {'level': 3, 'sources': ['other']}
    
    for sec in sections:
        sec_clean = sec.strip()
        if not sec_clean: continue
        
        header = sec_clean.split('\n')[0].lower()
        if any(k in header for k in ['experience', 'work', 'history', 'employment']):
            current_section = 'experience'
        elif 'skills' in header:
            current_section = 'skills'
        elif 'education' in header:
            current_section = 'education'
        elif 'projects' in header:
            current_section = 'projects'
        elif any(k in header for k in ['hobbies', 'interests']):
            current_section = 'hobbies'
        elif 'languages' in header:
            current_section = 'languages'
            
        # 2. Extract Skills with context
        sec_norm = normalize_text(sec_clean)
        for group, keywords in SKILL_GROUPS.items():
            for kw in keywords:
                if re.search(rf'\b{re.escape(kw)}\b', sec_norm):
                    if kw not in fields['skills_modeled']:
                        fields['skills_modeled'][kw] = {'level': 3, 'sources': []}
                    source = 'work' if current_section == 'experience' else 'projects' if current_section == 'projects' else 'other'
                    if source not in fields['skills_modeled'][kw]['sources']:
                        fields['skills_modeled'][kw]['sources'].append(source)
                    # Boost level if in work experience
                    if source == 'work': fields['skills_modeled'][kw]['level'] = 4

    # 3. Handle Detailed Sections
    total_months = 0
    for sec in sections:
        sec_clean = sec.strip()
        if not sec_clean: continue
        
        header = sec_clean.split('\n')[0].lower()
        if any(k in header for k in ['experience', 'work', 'history', 'employment']):
            all_experience_text += sec_clean + "\n"
            # Split by line but try to group role descriptions
            # Often roles are separated by double newlines or lines containing dates
            lines = [l.strip() for l in sec_clean.split('\n') if len(l.strip()) > 5]
            
            current_role = None
            for line in lines:
                # If line contains a date range, it's likely a new role header
                if extract_experience_from_dates(line) > 0:
                    if current_role:
                        fields['experience'].append(current_role)
                    current_role = {
                        'text': line,
                        'duration_months': extract_experience_from_dates(line),
                        'tools': []
                    }
                elif current_role:
                    current_role['text'] += " " + line
            
            if current_role:
                fields['experience'].append(current_role)
            
            # Extract skills from the whole section context
            for role in fields['experience']:
                role['tools'] = [kw for kw in fields['skills_modeled'] if kw in normalize_text(role['text'])]
                total_months += role['duration_months']
        elif 'projects' in header:
            proj_items = re.split(r'\n(?=project|name|\-|•)', sec_clean, flags=re.I)
            for p in proj_items: # Don't skip 0 if it contains info
                if len(normalize_text(p)) > 10:
                    fields['projects'].append({
                        'text': p.strip(),
                        'tools': [kw for kw in fields['skills_modeled'] if kw in normalize_text(p)]
                    })
        elif any(k in header for k in ['certif', 'diploma', 'licens']):
            # Extract bullet points
            items = re.split(r'\n|•|\-', sec_clean)
            fields['certifications'].extend([i.strip() for i in items if len(i.strip()) > 5 and 'certif' not in i.lower()])
        elif 'education' in header:
            fields['education'] += sec_clean + " "
        elif any(k in header for k in ['hobbies', 'interests']):
            items = re.split(r'[,;•\t]', sec_clean)
            fields['hobbies'].extend([i.strip() for i in items[1:] if i.strip()])
    
    # Calculate derived years from dates
    if total_months_from_dates > 0:
        fields['total_years'] = round(total_months_from_dates / 12.0, 1)
        fields['total_experience_str'] = format_duration(total_months_from_dates)
    else:
        # Try explicit mention at top level
        explicit = extract_explicit_durations(text)
        if explicit > 0:
            fields['total_years'] = round(explicit / 12.0, 1)
            fields['total_experience_str'] = format_duration(explicit)
        else:
            fields['total_years'] = extract_years(text)
            fields['total_experience_str'] = f"{fields['total_years']} years"

    fields['seniority'] = detect_seniority(text, fields['total_years'])
    fields['experience_context'] = all_experience_text # for semantic matching
    
    # Salary detection - Improved to avoid matching years
    # Matches values with k/000 and excludes common years (2020-2030)
    salary_match = re.search(r'(?:salary|expected|compensation|base)\s*[:\-–]?\s*(?:[\$€£]|mad)?\s*([1-9][\d,]*[0k])\b', text_norm, re.I)
    if salary_match:
        val_str = salary_match.group(1)
        if not (2020 <= int(re.sub(r'[^\d]', '', val_str.replace('k','000'))) <= 2035):
            fields['expected_salary'] = val_str

    return fields


def parse_cv_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        text = extract_text_from_pdf(path)
    elif ext in ['.docx', '.doc']:
        text = extract_text_from_docx(path)
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    fields = parse_structured_cv(text)
    cv = {
        'name': os.path.splitext(os.path.basename(path))[0],
        'total_years': fields.get('total_years', 0),
        'total_experience_str': fields.get('total_experience_str', '0 years'),
        'seniority': fields.get('seniority', 'Mid'),
        'certifications': fields.get('certifications', []),
        'skills_modeled': fields.get('skills_modeled', {}),
        'languages': fields['languages'],
        'soft_skills': fields.get('soft_skills', []),
        'work_preference': fields.get('work_preference', None),
        'education': fields['education'],
        'hobbies': fields['hobbies'],
        'expected_salary': fields['expected_salary'],
        'experience_context': fields['experience_context'],
        'projects': fields['projects'],
        'experience_list': fields['experience'],
        'text': text,
        'text_clean': preprocess(text)
    }
    return cv


# Parse an uploaded file-like object (e.g., Streamlit's UploadedFile)
import tempfile
import io

def parse_uploaded_file(uploaded_file):
    filename = getattr(uploaded_file, 'name', 'uploaded_cv')
    ext = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        data = uploaded_file.read()
        if hasattr(data, 'getvalue'):
            data = data.getvalue()
        if isinstance(data, str):
            data = data.encode('utf-8')
        tmp.write(data)
        tmp_path = tmp.name
    try:
        cv = parse_cv_file(tmp_path)
        cv['name'] = os.path.splitext(filename)[0]
        return cv
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def parse_job_posting(text):
    text_norm = normalize_text(text)
    
    # 1. Experience Requirements
    required_years = extract_required_years_from_job(text)
    seniority = detect_seniority(text, required_years)
    
    # 2. Sentence-level Skills Extraction
    # We split by common sentence terminators.
    sentences = re.split(r'[.!?\n;]+', text_norm)
    
    must_have = set()
    nice_to_have = set()
    
    # Triggers
    must_triggers = [
        'required', 'must', 'essential', 'strong', 'proficient', 'expert',
        'proven', 'hands-on', 'comfortable with', 'deep', 'solid', 'mandatory',
        'requirements', 'responsibilities'
    ]
    nice_triggers = [
        'plus', 'nice to have', 'preferred', 'desirable', 'advantage',
        'familiarity', 'optional', 'bonus', 'ideal'
    ]

    # Persist requirement mode across subsequent lines/bullets.
    # This helps when a posting says "nice to have:" once, then lists items.
    current_mode = None  # 'must' | 'nice' | None

    for sentence in sentences:
        sent_clean = sentence.strip()
        if len(sent_clean) < 5: continue
        
        # Determine sentiment of sentence and update persistent mode
        is_nice = any(t in sent_clean for t in nice_triggers)
        is_must = any(t in sent_clean for t in must_triggers)

        if is_nice and not is_must:
            current_mode = 'nice'
        elif is_must and not is_nice:
            current_mode = 'must'
        
        # Determine skills in this sentence
        found_skills = []
        for group, keywords in SKILL_GROUPS.items():
            for kw in keywords:
                # Use word boundary to avoid partial matches
                if re.search(rf'\b{re.escape(kw)}\b', sent_clean):
                    found_skills.append(kw)
        
        if not found_skills: continue
        
        if is_nice:
            nice_to_have.update(found_skills)
        elif is_must:
            must_have.update(found_skills)
        else:
            # Fallback: use persisted mode if available, else assume Must
            if current_mode == 'nice':
                nice_to_have.update(found_skills)
            else:
                must_have.update(found_skills)
            
    # 3. Language Requirements
    languages_required = infer_languages(text)

    # 3b. Soft skills requirements
    soft_skills_required = extract_soft_skills(text)
    
    # 4. Salary range
    salary_match = re.search(r'(?:salary|compensation|range|base|budget)\s*[:\-–]?\s*([\d,k\$\s-]+(?:mad|usd|eur)?)', text_norm, re.I)
    salary_range = salary_match.group(1).strip() if salary_match else "N/A"
    
    return {
        'required_years': required_years,
        'seniority': seniority,
        'must_have_skills': list(must_have),
        'nice_to_have_skills': list(nice_to_have),
        'languages_required': languages_required,
        'soft_skills_required': soft_skills_required,
        'work_mode': infer_work_mode(text),
        'salary_range': salary_range,
        'full_text': text,
        'full_text_clean': preprocess(text)
    }


def load_cvs_from_dir(folder='cvs'):
    if not os.path.exists(folder):
        return []
    files = glob.glob(os.path.join(folder, '*.*'))
    cvs_loaded = []
    for f in files:
        if os.path.splitext(f)[1].lower() in ['.pdf', '.docx', '.doc', '.txt']:
            try:
                cvs_loaded.append(parse_cv_file(f))
                print(f"Loaded CV from {f}")
            except Exception as e:
                print(f"Failed to parse {f}: {e}")
    return cvs_loaded

# Evaluation helpers

def build_synthetic_ground_truth(cvs, jobs_df):
    gt = {}
    for cv in cvs:
        skills = [s.lower() for s in cv.get('skills_modeled', {}).keys()]
        relevant = set()
        for i, row in jobs_df.iterrows():
            job_skills = str(row.get('job_skills', '')).lower()
            job_summary = str(row.get('job_summary', '')).lower()
            if any((skill in job_skills) or (skill in job_summary) for skill in skills):
                relevant.add(i)
        # Fallback: if no matches found by skills, try matching on job title / summary keywords (e.g., 'data engineer')
        if not relevant:
            for i, row in jobs_df.iterrows():
                jt = str(row.get('job_title', '')).lower()
                js = str(row.get('job_summary', '')).lower()
                if any((skill in jt) or ('data engineer' in jt) or (skill in js) or ('data engineer' in js) for skill in skills):
                    relevant.add(i)
        gt[cv['name']] = relevant
    return gt


def precision_at_k(results, ground_truth, k=5):
    precisions = {}
    for cv, scores in results.items():
        top_k_idx = np.argsort(scores)[::-1][:k]
        relevant = ground_truth.get(cv, set())
        hit = sum(1 for idx in top_k_idx if idx in relevant)
        precisions[cv] = hit / k
    mean_precision = float(np.mean(list(precisions.values()))) if precisions else 0.0
    return mean_precision, precisions

# Charger CVs supplémentaires si présents
extra_cvs = load_cvs_from_dir('cvs')
if extra_cvs:
    # éviter les doublons si mêmes noms
    existing_names = {c['name'] for c in cvs}
    for ecv in extra_cvs:
        if ecv['name'] in existing_names:
            ecv['name'] = ecv['name'] + '_file'
        cvs.append(ecv)

# Recompute text_clean in case we added new CVs
for cv in cvs:
    if 'text_clean' not in cv:
        cv['text_clean'] = preprocess(cv.get('text', ''))

# 4. TF-IDF + Cosine Similarity
def tfidf_matching(cvs, job_descriptions):
    results = {}
    # Use text_clean if available, else text
    texts = [cv['text_clean'] for cv in cvs] + list(job_descriptions)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    cv_vectors = tfidf_matrix[:len(cvs)]
    job_vectors = tfidf_matrix[len(cvs):]
    for i, cv in enumerate(cvs):
        scores = cosine_similarity(cv_vectors[i], job_vectors)[0]
        results[cv['name']] = scores
    return results

def run_comprehensive_matching(cv, jobs_df, weights=None, allow_project_compensation=True):
    """
    Runs the new weighted matching for a single CV against all jobs in jobs_df.
    Returns a DataFrame with scores and explanations.
    """
    # Canonical CV text
    cv_text_raw = cv.get('text', '')
    cv_text_clean = cv.get('text_clean', preprocess(cv_text_raw))

    # Build / reuse cached job artifacts
    artifacts = _get_job_artifacts(jobs_df, desc_col=job_text_col, tfidf_max_features=5000)
    n_jobs = len(artifacts['job_texts_raw'])

    # --- Stage 1: TF-IDF retrieval over all jobs (fast, high recall) ---
    try:
        vec = artifacts['tfidf_vectorizer']
        job_mat = artifacts['job_tfidf_matrix']
        cv_vec = vec.transform([cv_text_clean])
        tfidf_scores_all = cosine_similarity(cv_vec, job_mat).flatten()
    except Exception as e:
        print(f"TF-IDF retrieval failed: {e}")
        tfidf_scores_all = np.zeros(n_jobs, dtype=float)

    # Candidate pool size (adaptive): score everything for small datasets
    candidate_pool = 3000
    if n_jobs <= candidate_pool:
        candidate_indices = np.arange(n_jobs)
    else:
        # Pick top-N by TF-IDF
        candidate_indices = np.argsort(tfidf_scores_all)[::-1][:candidate_pool]

    # --- Stage 2: Deep/expensive scoring only on candidates ---
    cand_texts_raw = [artifacts['job_texts_raw'][i] for i in candidate_indices]
    cand_titles = [artifacts['job_titles'][i] for i in candidate_indices]

    # Embeddings for semantic matching
    try:
        model = _get_sentence_model('all-MiniLM-L6-v2')
        cv_embed = model.encode([cv_text_raw], show_progress_bar=False)
        cand_embeds = model.encode(cand_texts_raw, show_progress_bar=False)
        semantic_scores_cand = cosine_similarity(cv_embed, cand_embeds)[0]
    except Exception as e:
        print(f"Embedding failed: {e}")
        semantic_scores_cand = np.zeros(len(candidate_indices), dtype=float)

    # Title relevance (also embeddings, but only on candidates)
    try:
        title_embeds = model.encode(cand_titles, show_progress_bar=False)
        title_scores_cand = cosine_similarity(cv_embed, title_embeds)[0]
    except Exception as e:
        print(f"Title embedding failed: {e}")
        title_scores_cand = np.zeros(len(candidate_indices), dtype=float)

    # LDA topic similarity on candidates (stable: trained once on jobs)
    try:
        lda_model = artifacts.get('lda_model')
        lda_dictionary = artifacts.get('lda_dictionary')
        job_topic_vectors = artifacts.get('job_topic_vectors')
        lda_num_topics = int(artifacts.get('lda_num_topics', 5))

        if lda_model is None or lda_dictionary is None or job_topic_vectors is None:
            raise ValueError("Missing LDA artifacts")

        cv_tokens = simple_preprocess(cv_text_clean)
        cv_bow = lda_dictionary.doc2bow(cv_tokens)
        cv_topics = lda_model.get_document_topics(cv_bow, minimum_probability=0.0)
        cv_vec_topics = np.array([prob for _, prob in cv_topics], dtype=float)
        if cv_vec_topics.size != lda_num_topics:
            padded = np.zeros(lda_num_topics, dtype=float)
            padded[:min(lda_num_topics, cv_vec_topics.size)] = cv_vec_topics[:min(lda_num_topics, cv_vec_topics.size)]
            cv_vec_topics = padded

        lda_scores_cand = []
        for idx in candidate_indices:
            job_vec_topics = job_topic_vectors[int(idx)]
            lda_scores_cand.append(float(np.dot(cv_vec_topics, job_vec_topics)))
        lda_scores_cand = np.array(lda_scores_cand, dtype=float)

        # Normalize LDA similarity to [0,1] over candidates to keep weight meaningful
        if len(lda_scores_cand) > 0:
            mn, mx = float(lda_scores_cand.min()), float(lda_scores_cand.max())
            if mx > mn:
                lda_scores_cand = (lda_scores_cand - mn) / (mx - mn)
            else:
                lda_scores_cand = np.zeros_like(lda_scores_cand)
    except Exception as e:
        print(f"LDA failed: {e}")
        lda_scores_cand = np.zeros(len(candidate_indices), dtype=float)

    results = []
    for j, idx in enumerate(candidate_indices):
        idx = int(idx)
        job = _get_parsed_job(artifacts, idx)

        job_row = jobs_df.iloc[idx]

        tf_s = float(tfidf_scores_all[idx])
        lda_s = float(lda_scores_cand[j]) if j < len(lda_scores_cand) else 0.0
        sem_s = float(semantic_scores_cand[j]) if j < len(semantic_scores_cand) else 0.0

        score, explanation = attribute_weighted_matching(
            cv, job, weights,
            allow_project_compensation=allow_project_compensation,
            semantic_score=sem_s,
            tfidf_score=tf_s,
            lda_score=lda_s,
        )

        # Dynamic title adjustment
        t_score = float(title_scores_cand[j]) if j < len(title_scores_cand) else 0.0
        title_adjustment = 0.0
        if t_score > 0.4:
            title_adjustment = 0.15 * t_score
        elif t_score < 0.15:
            title_adjustment = -0.05

        final_score = max(0.0, min(1.0, float(score) + float(title_adjustment)))

        # Enforce must-have gate after any post-adjustments
        gate_cap = float(explanation.get('gate_cap', 1.0))
        final_score = min(final_score, gate_cap)

        explanation['final_score'] = final_score
        explanation['title_relevance'] = f"{t_score:.2f}"
        explanation['breakdown']['Role Match'] = float(title_adjustment)

        results.append({
            'job_index': idx,
            'job_title': job_row.get('job_title', 'N/A'),
            'company': job_row.get('company', ''),
            'job_location': job_row.get('job_location', ''),
            'job_link': job_row.get('job_link', ''),
            'first_seen': job_row.get('first_seen', ''),
            'job_level': job_row.get('job level', job_row.get('job_level', '')),
            'job_type': job_row.get('job_type', ''),
            'job_summary': job_row.get('job_summary', ''),
            'job_skills': job_row.get('job_skills', ''),
            'job_text': job_row.get('job_text', ''),
            'must_have_skills': job.get('must_have_skills', []),
            'nice_to_have_skills': job.get('nice_to_have_skills', []),
            'languages_required': job.get('languages_required', []),
            'salary_range': job.get('salary_range', 'N/A'),
            'score': final_score,
            'explanation': explanation,
            'seniority': job.get('seniority', 'Mid'),
            'required_years': job.get('required_years', 0.0),
        })

    return pd.DataFrame(results).sort_values('score', ascending=False)

def calculate_advanced_skills_match(cv_skills, job_must_have, job_nice_to_have):
    # cv_skills is {skill: {level, sources}}
    matched_must = []
    missing_must = []
    matched_nice = []
    
    cv_set = set(cv_skills.keys())
    
    for s in job_must_have:
        if s in cv_set: matched_must.append(s)
        else: missing_must.append(s)
        
    for s in job_nice_to_have:
        if s in cv_set: matched_nice.append(s)
        
    must_coverage = len(matched_must) / len(job_must_have) if job_must_have else 1.0
    nice_coverage = len(matched_nice) / len(job_nice_to_have) if job_nice_to_have else 1.0
    
    return must_coverage, nice_coverage, matched_must, missing_must, matched_nice

def calculate_language_gate(cv_langs, job_reqs):
    if not job_reqs: return 1.0, []
    
    cv_map = {l['language'].lower(): l['level'] for l in cv_langs}
    gaps = []
    total_penalty = 0
    
    for req in job_reqs:
        lang = req['language'].lower()
        min_level = req.get('level', 2) # Assume Intermediate if not specified
        
        cv_level = cv_map.get(lang, 0)
        if cv_level < min_level:
            penalty = (min_level - cv_level) * 0.2
            total_penalty += penalty
            gaps.append(f"{req['language']} level {cv_level} < {min_level}")
            
    score = max(0.0, 1.0 - total_penalty)
    return score, gaps

def compare_seniority(cv_seniority, job_seniority):
    levels = ['Junior', 'Mid', 'Senior', 'Lead']
    try:
        cv_idx = levels.index(cv_seniority)
        job_idx = levels.index(job_seniority)
    except ValueError:
        return 0.5
    
    if cv_idx >= job_idx:
        return 1.0
    elif cv_idx == job_idx - 1:
        return 0.7
    else:
        return 0.3

def attribute_weighted_matching(cv, job, weights=None, allow_project_compensation=True, semantic_score=0, tfidf_score=0, lda_score=0):
    if weights is None:
        # Balanced weights for Hybrid Approach including Education/Certs
        weights = {
            'skills': 0.30,      # Core competency
            'experience': 0.15,  # History
            'languages': 0.05,   # Gate
            'qualifications': 0.10, # Education + Certs
            'constraints': 0.05, # Salary
            'soft_skills': 0.08, # Hiring signals
            # NLP Factors (Total 0.35)
            'responsibilities': 0.15, # Semantic Embedding
            'tfidf': 0.10,            # Keyword Overlap
            'lda': 0.10               # Topic Fit
        }
    
    # Normalize weights to sum to 1.0
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    
    # 1. Skills Match
    must_cov, nice_cov, m_must, miss_must, m_nice = calculate_advanced_skills_match(
        cv.get('skills_modeled', {}), job['must_have_skills'], job['nice_to_have_skills']
    )

    # Must-have gating (critical)
    if job['must_have_skills']:
        if must_cov < 0.25:
            gate_cap = 0.35
        elif must_cov < 0.50:
            gate_cap = 0.55
        else:
            gate_cap = 1.0
    else:
        gate_cap = 1.0
    # Skill score: Must-haves are 80%, Nice-to-haves are 20%
    skills_score = (must_cov * 0.8) + (nice_cov * 0.2)
    
    # 2. Project Compensation Logic
    project_boost = 0
    gaps_filled_by_projects = []
    if allow_project_compensation:
        for missing in miss_must:
            # Check if this skill is in a project
            for p in cv['projects']:
                if missing in p.get('tools', []):
                    project_boost += 0.05 # Small boost for each missing must-have found in projects
                    gaps_filled_by_projects.append(missing)
                    break
    
    # 3. Experience Match
    exp_score = compare_seniority(cv['seniority'], job['seniority'])
    # Penalty if years are way off
    if cv['total_years'] < job['required_years']:
        exp_score *= (cv['total_years'] / job['required_years']) if job['required_years'] > 0 else 1.0
        
    # 4. Languages & Certifications
    lang_score, lang_gaps = calculate_language_gate(cv['languages'], job['languages_required'])
    
    relevant_certs = 0
    all_job_skills = set(job['must_have_skills'] + job['nice_to_have_skills'])
    if cv.get('certifications'):
        for cert in cv['certifications']:
            if any(skill in cert.lower() for skill in all_job_skills):
                relevant_certs += 1
    
    # Education Matching
    edu_score = 0.5 # Baseline
    cv_edu = cv.get('education', '').lower()
    if any(k in cv_edu for k in ['bachelor', 'degree', 'engineer', 'bs', 'license']):
        edu_score = 0.8
    if any(k in cv_edu for k in ['master', 'phd', 'ms', 'doctorate']):
        edu_score = 1.0
        
    # Combined Qualifications Score
    qual_score = min(1.0, edu_score + (relevant_certs * 0.1))

    # 5. Salary fit
    salary_score = 1.0
    if cv['expected_salary'] and job['salary_range'] != "N/A":
        try:
            cv_val = int(re.sub(r'[^\d]', '', cv['expected_salary'].replace('k', '000')))
            job_vals = [int(v) for v in re.findall(r'\d+', job['salary_range'].replace('k', '000'))]
            if job_vals:
                if cv_val <= max(job_vals) * 1.1: salary_score = 1.0
                else: salary_score = 0.5
        except: salary_score = 0.8

    # 5b. Work mode preference fit (Remote/Hybrid/Onsite) — only if detectable
    work_mode_score = None
    cv_pref = cv.get('work_preference', None)
    job_mode = job.get('work_mode', '')
    if cv_pref and job_mode:
        if cv_pref == job_mode:
            work_mode_score = 1.0
        elif cv_pref == 'Remote' and job_mode == 'Hybrid':
            work_mode_score = 0.6
        elif cv_pref == 'Hybrid' and job_mode == 'Remote':
            work_mode_score = 0.85
        elif cv_pref == 'Onsite' and job_mode == 'Hybrid':
            work_mode_score = 0.85
        else:
            work_mode_score = 0.2

    # Combine constraints components if available; otherwise keep it a no-op
    constraints_score = salary_score
    if work_mode_score is not None and cv['expected_salary'] and job['salary_range'] != "N/A":
        constraints_score = (0.6 * salary_score) + (0.4 * float(work_mode_score))
    elif work_mode_score is not None:
        constraints_score = float(work_mode_score)

    # 6. Soft skills (structured)
    soft_score = 1.0
    matched_soft, missing_soft = [], []
    job_soft = job.get('soft_skills_required', [])
    if job_soft:
        cv_soft = set(cv.get('soft_skills', []))
        matched_soft = [s for s in job_soft if s in cv_soft]
        missing_soft = [s for s in job_soft if s not in cv_soft]
        soft_score = len(matched_soft) / max(1, len(job_soft))
        
    # Combined final score
    final_score = (
        weights['skills'] * skills_score +
        weights['experience'] * exp_score +
        weights['responsibilities'] * semantic_score +
        weights['languages'] * lang_score +
        weights.get('qualifications', 0.10) * qual_score +
        weights['constraints'] * constraints_score +
        weights.get('tfidf', 0.10) * tfidf_score +
        weights.get('lda', 0.10) * lda_score +
        weights.get('soft_skills', 0.0) * soft_score
    ) + project_boost

    final_score = min(final_score, gate_cap)
    
    # Explanation
    explanation = {
        'why_matched': [
            f"Matched {len(m_must)}/{len(job['must_have_skills'])} must-have skills",
            f"Experience: {cv['total_years']}y vs {job['required_years']}y required",
        ],
        'gaps': [f"Missing must-have: {s}" for s in miss_must if s not in gaps_filled_by_projects],
        'project_highlights': [f"Project covered missing: {s}" for s in gaps_filled_by_projects],
        'language_status': "Met" if lang_score > 0.9 else "Requirement Gap",
        'gate_cap': float(gate_cap),
        'final_score': float(final_score),
        'breakdown': {
            'Skills ': float(weights['skills'] * skills_score),
            'Experience ': float(weights['experience'] * exp_score),
            'Education/Certs ': float(weights.get('qualifications', 0.1) * qual_score),
            'Semantics ': float(weights['responsibilities'] * semantic_score),
            'TF-IDF ': float(weights.get('tfidf', 0.1) * tfidf_score),
            'LDA ': float(weights.get('lda', 0.1) * lda_score),
            'Languages ': float(weights['languages'] * lang_score),
            'Salary ': float(weights['constraints'] * constraints_score),
            'Soft skills ': float(weights.get('soft_skills', 0.0) * soft_score),
            'Projects ': float(project_boost)
        }
    }

    if cv_pref and job_mode:
        explanation['why_matched'].append(f"Work mode: CV prefers {cv_pref}, job is {job_mode}")

    explanation['why_matched'].append(
        f"Soft skills matched: {', '.join(matched_soft[:5])}" if matched_soft else "Soft skills: none detected"
    )
    if missing_soft:
        explanation['gaps'].extend([f"Missing soft skill: {s}" for s in missing_soft[:5]])
    if lang_gaps: explanation['gaps'].extend(lang_gaps)
    
    return final_score, explanation

def lda_matching(cvs, job_descriptions, num_topics=5):
    all_texts = [cv['text_clean'] for cv in cvs] + list(job_descriptions)
    tokenized = [simple_preprocess(t) for t in all_texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    def get_topic_vector(text):
        bow = dictionary.doc2bow(simple_preprocess(text))
        topic_dist = lda.get_document_topics(bow, minimum_probability=0)
        return np.array([prob for _, prob in topic_dist])
    cv_topics = [get_topic_vector(cv['text_clean']) for cv in cvs]
    job_topics = [get_topic_vector(desc) for desc in job_descriptions]
    results = {}
    for i, cv_vec in enumerate(cv_topics):
        # Use dot product as similarity measure on topic distributions
        scores = [float(np.dot(cv_vec, job_vec)) for job_vec in job_topics]
        results[cvs[i]['name']] = np.array(scores)
    return results


def embedding_matching(cvs, job_descriptions, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    cv_embeds = model.encode([cv['text_clean'] for cv in cvs], show_progress_bar=False)
    job_embeds = model.encode(list(job_descriptions), show_progress_bar=False)
    results = {}
    for i, cv_vec in enumerate(cv_embeds):
        scores = cosine_similarity([cv_vec], job_embeds)[0]
        results[cvs[i]['name']] = np.array(scores)
    return results


def rank_results(results, jobs, top_n=3):
    ranking = {}
    for cv_name, scores in results.items():
        scores = np.array(scores, dtype=float)
        top_indices = np.argsort(scores)[::-1][:top_n]
        ranking[cv_name] = [(
            jobs.iloc[i].get('job_title') if 'job_title' in jobs.columns else jobs.iloc[i].get('title', 'N/A'),
            float(scores[i])
        ) for i in top_indices]
    return ranking


def normalize_scores_array(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def ensemble_matching(tfidf_res, lda_res, embed_res, weights=(1.0, 1.0, 1.0)):
    results = {}
    for cv in tfidf_res.keys():
        tf = normalize_scores_array(tfidf_res[cv])
        ld = normalize_scores_array(lda_res[cv])
        em = normalize_scores_array(embed_res[cv])
        combined = (weights[0] * tf + weights[1] * ld + weights[2] * em) / sum(weights)
        results[cv] = np.array(combined)
    return results


def results_to_df(results, jobs):
    rows = []
    for cv, scores in results.items():
        for i, s in enumerate(scores):
            rows.append({
                'cv': cv,
                'job_index': int(i),
                'job_title': jobs.iloc[i].get('job_title') if 'job_title' in jobs.columns else jobs.iloc[i].get('title', 'N/A'),
                'score': float(s)
            })
    return pd.DataFrame(rows).sort_values(['cv', 'score'], ascending=[True, False])

# Delay heavy computation into a main() so file can be imported safely

def run_pipeline(save_outputs=True):
    # Unified pipeline: use the same hybrid matcher as the Streamlit app
    all_results = {}
    for cv_obj in cvs:
        name = cv_obj.get('name', 'cv')
        print(f"\nRunning unified matching for {name}...")
        df_res = run_comprehensive_matching(cv_obj, df_jobs)
        all_results[name] = df_res
        if save_outputs:
            df_res.to_csv(f"matching_results_{name}.csv", index=False)
            print(f"Saved detailed results to matching_results_{name}.csv")
            df_res.head(5).to_csv(f"top_matches_{name}.csv", index=False)
            print(f"Saved top matches to top_matches_{name}.csv")

    # === Evaluation: precision@k using a synthetic ground truth ===
    ground_truth = build_synthetic_ground_truth(cvs, df_jobs)
    for cv_name, rel in ground_truth.items():
        print(f"Ground truth for {cv_name}: {len(rel)} relevant jobs")

    # Convert per-CV ranked results into score arrays for precision_at_k
    results_for_eval = {}
    for cv_obj in cvs:
        name = cv_obj.get('name', 'cv')
        scores_array = np.zeros(len(df_jobs), dtype=float)
        df_res = all_results.get(name)
        if df_res is not None and not df_res.empty:
            for _, r in df_res.iterrows():
                scores_array[int(r['job_index'])] = float(r['score'])
        results_for_eval[name] = scores_array

    eval_rows = []
    for k in [1, 3, 5]:
        mean_p, per_cv = precision_at_k(results_for_eval, ground_truth, k=k)
        eval_rows.append({'algorithm': 'unified', 'k': k, 'mean_precision': mean_p})
        if save_outputs:
            per_df = pd.DataFrame([{'cv': cv, f'precision@{k}': prec} for cv, prec in per_cv.items()])
            per_df.to_csv(f'precision_unified_k{k}.csv', index=False)
        print(f'Precision@{k} for unified: {mean_p:.4f}')

    df_eval = pd.DataFrame(eval_rows)
    if save_outputs:
        df_eval.to_csv('evaluation_precision_at_k.csv', index=False)
        print('Saved evaluation summary to evaluation_precision_at_k.csv')

    return {'unified_results': all_results, 'eval_summary': df_eval}


# --- Petit protocole de test synthétique ---
def run_synthetic_test():
    print('\nRunning synthetic test protocol...')
    jobs = pd.DataFrame([
        {'job_title':'Python Data Engineer', 'job_summary':'Looking for python and aws','job_skills':'Python, AWS'},
        {'job_title':'Spark Data Engineer', 'job_summary':'Spark and Hadoop experience','job_skills':'Spark, Hadoop'},
        {'job_title':'Java Engineer', 'job_summary':'Java backend','job_skills':'Java'},
        {'job_title':'Data Scientist', 'job_summary':'ML and python','job_skills':'Python, ML'},
        {'job_title':'ML Engineer', 'job_summary':'Docker, MLops','job_skills':'Docker, ML'},
        {'job_title':'Junior Data Engineer','job_summary':'entry level python','job_skills':'Python'}
    ])
    cvs_small = [
        {'name':'Synth_A', 'skills':['Python','AWS'], 'text': 'Skills: Python AWS.', 'text_clean': preprocess('Skills: Python AWS.')},
        {'name':'Synth_B', 'skills':['Spark'], 'text': 'Skills: Spark.', 'text_clean': preprocess('Skills: Spark.')},
        {'name':'Synth_C', 'skills':['Ruby'], 'text': 'Skills: Ruby.', 'text_clean': preprocess('Skills: Ruby.')}
    ]
    tf = tfidf_matching(cvs_small, jobs['job_summary'].apply(preprocess))
    lda = lda_matching(cvs_small, jobs['job_summary'].apply(preprocess), num_topics=3)
    emb = embedding_matching(cvs_small, jobs['job_summary'].apply(preprocess))
    gt = {'Synth_A': {0,5}, 'Synth_B': {1}, 'Synth_C': set()}
    synth_results = []
    for name,res in [('TF-IDF',tf),('LDA',lda),('Embed',emb)]:
        for k in [1,3,5]:
            mean, per = precision_at_k(res, gt, k=k)
            print(f'SYNTH {name} Precision@{k}: {mean:.3f}')
            synth_results.append({'algorithm': name, 'k': k, 'mean_precision': mean})
    pd.DataFrame(synth_results).to_csv('synthetic_test_results.csv', index=False)
    print('Saved synthetic test summary to synthetic_test_results.csv')


if __name__ == '__main__':
    # Run full pipeline when executed directly
    run_pipeline()

