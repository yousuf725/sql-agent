import os
import json 
import sqlite3
import tempfile
from pathlib import Path 
import streamlit  as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# config

DEFAULT_MODEL = "llama-3.1-8b-instant"

MAX_ROWS = 20

# LLM

@st.cache_resource
def get_llm(groq_api_key: str | None):
    '''
    Create (and cache) a ChatGroq instance.
    Read key from argument or GROQ_API_KEY env.
    '''

    key = groq_api_key or os.getenv("GROQ_API_KEY")

    if not key:
        raise ValueError("Groq API Key not provided. Set in sidebar or env GROQ_API_KEY")
    
    return ChatGroq(
        model = DEFAULT_MODEL,
        temperature = 0,
        streaming= False,
        api_key = key
    )

def save_uploaded_db(uploaded_file) -> str:
    """ Save uploaaded .db file to a temp path and return the path."""

    t = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    t.write(uploaded_file.read())
    t.flush()
    return t.name
    

def get_db_path(uploaded_file) -> str | None:
    '''Decide which DB file to use:
    - if upload present -> temp file
    - else if e3_student.db exists longside the script 
    - else none
    '''
    if uploaded_file is not None:
        return save_uploaded_db(uploaded_file)
    
    default_db = Path(__file__).parent / "e3_student.db"

    if default_db.exists():
        return str(default_db)
    
    return None 

def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row 
    return conn


def get_schema(conn: sqlite3.Connection) -> dict:
    schema = {}

    cur = conn.cursor()

    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ).fetchall()

    for (table_name,) in tables:
        cols = cur.execute(f"PRAGMA table_info({table_name});").fetchall()

        col_names = [c[1] for c in cols]

        schema[table_name] = col_names
    return schema

def schema_to_text(schema: dict) -> str:
    """Format schema dict into  compact text block for the prompt."""
    lines = []

    for table, cols in schema.items():
        preview = ", ".join(cols[:10])
        extra = " ..." if len(cols)> 10 else ""
        lines.append(f"- {table}({preview}{extra})")
    return "\n".join(lines)

# agent logic

def ask_llm_for_sql(llm: ChatGroq, question: str, schema_text: str)-> dict:
    '''
    Ask LLM to propose a safe select query.
    
    Return dict
    {
    "sql": "...},
    "thiking" : "...",
    "followups": ["...","..."]
    }
    '''
    system = SystemMessage(
        content= (
            "You are 'DataGenie', a helpful SQL expert for SQLite database.\n"
            "You Must use only the tables and columns listed in SCHEMA below.\n"
            "Write ONLY safe SELECT queries (no INSERT/UPDATE/DELETE, no PRAGMA, no DROP, etc.).\n "
            "If the questionn is vague, make a reasonnable assumption mention it in 'thinking'.\n"
            "ALWAYS add a LIMIT clause (e.g., LIMIT 20) if user does not specify one.\n"
            "Return your answer as strict JSON with keys:  sql, thinking, followups.\n"
            "followups = a short list of 2 extra questions the user might like.\n"
            f"SCHEMA:\n {schema_text}"
        )
    )
    # we create a HumanMessage that passes the user's question and force the JSON output format
    user = HumanMessage(
        content = (
            f"User question: {question}\n\n"
            "Reply ONLY in JSON like this:\n"
            '{"sql":"...","thinking":"...","followups":["...","..."]}'
        )
    ) 
    resp = llm.invoke([system, user])
    text = resp.content.strip()

    try:
        start = text.index("{")
        end = text.rindex("}")+ 1
        json_text = text[start:end]
        data = json.loads(json_text)
    except Exception:
        data = {
            "sql": "SELECT 'Sorry, I could not generate SQL' AS error",
            "thinking": "I failed to follow my own JSON format.",
            "followups" : [
                "Try asking a simpler question.",
                "Ask me what tables exist and what they contain."
            ],
        }
    data.setdefault("thinking","")
    data.setdefault("followups",[])

    if not isinstance(data["followups"],list):
        data["followups"] = [str(data["followups"])]

    return data

def run_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame | str:
    """
    Execute the propose SQL safely.
    Returns:
        - DataFrame with up to MAX_ROWS rows, or 
        - error message string
    """
    sql_clean = sql.strip().rstrip(";")

    if not sql_clean.lower().startswith("select"):
        return "Blocked: Only SELECT queries are allowed"
    
    if "limit" not in sql_clean.lower():
        sql_to_run = f"{sql_clean} LIMIT {MAX_ROWS}"
    else:
        sql_to_run = sql_clean
    
    try:
        df = pd.read_sql_query(sql_to_run, conn)

        return df
    except Exception as e:
        return f"SQL Error: {e}"

def build_final_answer(llm: ChatGroq, question: str, sql: str, result) -> str:
    """
    Ask LLM to explain the result in a friendly way.
    `result` is either DataFrame or error string.
    """
    if isinstance(result, pd.DataFrame):
        if result.empty:
            result_text = "The query returnned 0 rows."
        else:
            preview = result.head(min(5, len(result)))
            result_text = "Here is a preview of the results (up to 5 rows):\n"
            result_text += preview.to_markdown(index=False)
    else:
        result_text = str(result)
    
    system = SystemMessage(
        content= (
            "You are 'DataGenie', an AI tutor.\n"
            "Explain what the SQL result means in simple, encouraging language.\n"
            "If there was an error, explain it gently annd hint how to fix the query.\n"
            "End with one short playful line (e.g about being a data genie)."
        )
    )
    user = HumanMessage(
        content = f"User question: {question}\nSQL used:\n{sql}\n\nResult Summary:\n{result_text}"
    )

    resp = llm.invoke([system, user])

    return resp.content.strip()


   # STREAMLIT APP 

def main():
    st.set_page_config(
        page_title="DataGenie",
        page_icon="🧞‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧞‍♂️ DataGenie: Talk to your SQLite Database")
    st.markdown(
        """
        - Upload a `.db` file (or keep a `e3_student.db` next to this script).  
        - Ask questions in **English or Roman Urdu**.  
        - See the **exact SQL query**, **results**, and **smart follow-up suggestions**.  
        - Watch how the *DataGenie* thinks about your question. ✨
        """
    )

    # Sidebar: DB + API key
    with st.sidebar:
        st.header("Step 1: Database")
        uploaded = st.file_uploader("Upload SQLite .db", type=["db", "sqlite"])
        st.caption("If you don't upload, I'll look for `e3_student.db` in this folder.")

        st.header("Step 2: Groq API Key")
        key_input = st.text_input(
            "GROQ_API_KEY",
            type="password",
            help="Get it from console.groq.com → API Keys",
        )
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input

        st.markdown("---")
        st.write("Made for teaching: see the SQL, see the results, see the 'Genie brain'. 💡")

    # Decide DB path
    db_path = get_db_path(uploaded)
    if not db_path:
        st.warning("No database available. Please upload a `.db` file or add `e3_student.db` next to this script.")
        return

    # Connect DB
    try:
        conn = connect_db(db_path)
    except Exception as e:
        st.error(f"Could not open database: {e}")
        return

    # Get schema
    schema = get_schema(conn)
    if not schema:
        st.error("No user tables found in this database.")
        return

    st.subheader("📚 Detected Tables & Columns")
    st.code(schema_to_text(schema))

    # Prepare LLM
    try:
        llm = get_llm(os.getenv("GROQ_API_KEY"))
    except Exception as e:
        st.error(str(e))
        return

    # Chat history
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Render previous turns
    for turn in st.session_state["history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # User input
    user_q = st.chat_input("Ask DataGenie about your data...")

    if user_q:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_q)
        st.session_state["history"].append({"role": "user", "content": user_q})

        # Agent response
        with st.chat_message("assistant"):
            st.markdown("🤔 **DataGenie is reading your schema and cooking up SQL...**")

            schema_text = schema_to_text(schema)
            plan = ask_llm_for_sql(llm, user_q, schema_text)

            sql = plan.get("sql", "")
            thinking = plan.get("thinking", "")
            followups = plan.get("followups", [])[:3]

            # Thought bubble + SQL
            if thinking:
                st.markdown(f"🧠 **Agent's thought bubble:** {thinking}")
            if sql:
                st.markdown("**Generated SQL:**")
                st.markdown(f"```sql\n{sql}\n```")
            else:
                st.warning("No SQL was generated.")

            # Run SQL
            result = run_sql(conn, sql) if sql else "No SQL to run."

            # Show results
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    st.info("Query ran successfully but returned **0 rows**.")
                else:
                    st.dataframe(result, use_container_width=True)
            else:
                if result.startswith("SQL Error") or result.startswith("Blocked"):
                    st.error(result)
                else:
                    st.write(result)

            # Natural language explanation
            final_answer = build_final_answer(llm, user_q, sql, result)
            st.markdown("---")
            st.markdown(final_answer)

            # Fun follow-up buttons
            if followups:
                st.markdown("**Do you also want to know:**")
                cols = st.columns(len(followups))
                for i, fq in enumerate(followups):
                    if cols[i].button(f"👉 {fq}"):
                        st.session_state["history"].append({"role": "user", "content": fq})
                        st.experimental_rerun()

        # Store last assistant message in history
        st.session_state["history"].append(
            {"role": "assistant", "content": final_answer}
        )


if __name__ == "__main__":
    main()