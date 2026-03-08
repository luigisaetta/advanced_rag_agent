"""
File name: core/post_answer_feedback.py
Author: Luigi Saetta
Last modified: 06-03-2026
Python Version: 3.11

Description:
    This module implements persistence of PostAnswerEvaluator results.
"""

from datetime import datetime, date, timedelta
import json
import oracledb

from core.utils import get_console_logger
from config_private import CONNECT_ARGS

logger = get_console_logger()


class PostAnswerFeedback:
    """
    Register post-answer evaluator results in Oracle DB.
    """

    TABLE_NAME = "POST_ANSWER_EVALUATION"
    CONFIDENCE_COLUMN = "CONFIDENCE"

    def get_connection(self):
        """Establish the Oracle DB connection."""
        return oracledb.connect(**CONNECT_ARGS)

    def table_exists(self, table_name: str) -> bool:
        """
        Check that the table exists in the current schema.
        """
        sql = """
            SELECT COUNT(*)
            FROM user_tables
            WHERE table_name = :tn
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, tn=table_name.upper())
                return cursor.fetchone()[0] > 0

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """
        Check whether a column exists for a table in the current schema.
        """
        sql = """
            SELECT COUNT(*)
            FROM user_tab_columns
            WHERE table_name = :tn
              AND column_name = :cn
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    tn=table_name.upper(),
                    cn=column_name.upper(),
                )
                return cursor.fetchone()[0] > 0

    def _create_table(self):
        """
        Create the table if it doesn't exist.
        """
        logger.info("Creating table %s...", self.TABLE_NAME)

        ddl_instr = f"""
        CREATE TABLE {self.TABLE_NAME} (
        ID         NUMBER GENERATED ALWAYS AS IDENTITY
                    (START WITH 1 INCREMENT BY 1) PRIMARY KEY,
        CREATED_AT DATE        DEFAULT SYSDATE NOT NULL,
        QUESTION   CLOB                        NOT NULL,
        ROOT_CAUSE VARCHAR2(30)                NOT NULL,
        REASON     CLOB                        NOT NULL,
        CONFIDENCE NUMBER(5,4),
        CONFIG_JSON CLOB                       NOT NULL
        )
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl_instr)

    def _ensure_schema(self) -> None:
        """
        Ensure table and required columns exist (non-destructive migration).
        """
        if not self.table_exists(self.TABLE_NAME):
            logger.info("Table %s doesn't exist...", self.TABLE_NAME)
            self._create_table()
            return

        if not self.column_exists(self.TABLE_NAME, self.CONFIDENCE_COLUMN):
            logger.info(
                "Adding column %s to %s...",
                self.CONFIDENCE_COLUMN,
                self.TABLE_NAME,
            )
            alter_sql = f"""
                ALTER TABLE {self.TABLE_NAME}
                ADD ({self.CONFIDENCE_COLUMN} NUMBER(5,4))
            """
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(alter_sql)

    def insert_feedback(
        self,
        question: str,
        root_cause: str,
        reason: str,
        confidence: float | None,
        config_data: dict,
    ):
        """Insert a new post-answer evaluator record."""
        sql = f"""
            INSERT INTO {self.TABLE_NAME}
            (CREATED_AT, QUESTION, ROOT_CAUSE, REASON, CONFIDENCE, CONFIG_JSON)
            VALUES (:created_at, :question, :root_cause, :reason, :confidence, :config_json)
        """

        self._ensure_schema()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                sql,
                {
                    "created_at": datetime.now(),
                    "question": str(question or "").strip(),
                    "root_cause": str(root_cause or "").strip(),
                    "reason": str(reason or "").strip(),
                    "confidence": confidence,
                    "config_json": json.dumps(config_data or {}, ensure_ascii=True),
                },
            )
            conn.commit()
            cursor.close()

    @staticmethod
    def _read_lob_if_needed(value):
        """
        Convert Oracle LOB values to plain strings.
        """
        if value is None:
            return ""
        if hasattr(value, "read"):
            try:
                return value.read()
            except Exception:
                return str(value)
        return str(value)

    def list_feedback(
        self,
        max_rows: int = 200,
        root_cause: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        llm_model_id: str | None = None,
    ) -> list[dict]:
        """
        Return post-answer evaluation rows ordered by newest first.
        Optional filters:
        - root_cause exact match
        - date_from/date_to inclusive range (date portion of CREATED_AT)
        - llm_model_id exact match from CONFIG_JSON.llm_model_id
          (fallback to CONFIG_JSON.model_id for backward compatibility)
        """
        if not self.table_exists(self.TABLE_NAME):
            return []
        self._ensure_schema()

        max_rows = max(1, min(int(max_rows), 2000))
        where_clauses = []
        binds = {"max_rows": max_rows}

        if root_cause:
            where_clauses.append("ROOT_CAUSE = :root_cause")
            binds["root_cause"] = str(root_cause).strip().upper()

        if date_from:
            where_clauses.append("CREATED_AT >= :date_from")
            binds["date_from"] = datetime.combine(date_from, datetime.min.time())

        if date_to:
            # inclusive on selected day
            next_day = date_to + timedelta(days=1)
            where_clauses.append("CREATED_AT < :date_to_next")
            binds["date_to_next"] = datetime.combine(next_day, datetime.min.time())

        if llm_model_id:
            where_clauses.append(
                """
                COALESCE(
                    JSON_VALUE(CONFIG_JSON, '$.llm_model_id' RETURNING VARCHAR2(200) NULL ON ERROR),
                    JSON_VALUE(CONFIG_JSON, '$.model_id' RETURNING VARCHAR2(200) NULL ON ERROR)
                ) = :llm_model_id
                """
            )
            binds["llm_model_id"] = str(llm_model_id).strip()

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT * FROM (
                SELECT
                    ID,
                    CREATED_AT,
                    QUESTION,
                    ROOT_CAUSE,
                    REASON,
                    CONFIDENCE,
                    CONFIG_JSON
                FROM {self.TABLE_NAME}
                {where_sql}
                ORDER BY CREATED_AT DESC
            )
            WHERE ROWNUM <= :max_rows
        """

        rows_out = []
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, binds)
                for row in cursor.fetchall():
                    cfg_raw = self._read_lob_if_needed(row[6])
                    try:
                        cfg_obj = json.loads(cfg_raw) if cfg_raw else {}
                    except Exception:
                        cfg_obj = {}

                    rows_out.append(
                        {
                            "id": int(row[0]),
                            "created_at": row[1],
                            "question": self._read_lob_if_needed(row[2]),
                            "root_cause": str(row[3] or ""),
                            "reason": self._read_lob_if_needed(row[4]),
                            "confidence": (
                                None if row[5] is None else float(row[5])
                            ),
                            "llm_model_id": (
                                (cfg_obj.get("llm_model_id") or cfg_obj.get("model_id"))
                                if isinstance(cfg_obj, dict)
                                else ""
                            ),
                            "config_json": cfg_obj,
                        }
                    )

        return rows_out
