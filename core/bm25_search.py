"""
File name: core/bm25_search.py
Author: Luigi Saetta
Last modified: 25-02-2026
Python Version: 3.11

License: MIT

Description:
    BM25 search engine implementation that indexes text retrieved from Oracle Database.
"""

import re
import oracledb
import numpy as np
from rank_bm25 import BM25Okapi
from core.utils import get_console_logger
from core.db_utils import get_connection

logger = get_console_logger()


class BM25OracleSearch:
    """
    Implments BM25
    """

    def __init__(self, table_name, text_column, batch_size=40):
        """
        Initializes the BM25 search engine with data from an Oracle database.

        :param table_name: Name of the table containing the text
        :param text_column: Column name that contains the text to index
        """
        self.table_name = table_name
        self.text_column = text_column
        self.batch_size = batch_size
        self.docs = []
        self.texts = []
        self.tokenized_texts = []
        self.bm25 = None
        self.index_data()

    @classmethod
    def from_serialized_payload(cls, payload: dict) -> "BM25OracleSearch":
        """
        Recreate an engine from serialized data without DB access.
        """
        obj = cls.__new__(cls)
        obj.table_name = payload["table_name"]
        obj.text_column = payload["text_column"]
        obj.batch_size = int(payload.get("batch_size", 40))
        obj.docs = payload.get("docs", [])
        obj.texts = payload.get("texts", [])
        obj.tokenized_texts = payload.get("tokenized_texts", [])
        obj.bm25 = BM25Okapi(obj.tokenized_texts) if obj.tokenized_texts else None
        return obj

    def to_serialized_payload(self) -> dict:
        """
        Return a serializable payload for cache persistence.
        """
        return {
            "table_name": self.table_name,
            "text_column": self.text_column,
            "batch_size": self.batch_size,
            "docs": self.docs,
            "texts": self.texts,
            "tokenized_texts": self.tokenized_texts,
        }

    @staticmethod
    def _validate_identifier(identifier: str) -> str:
        """
        Validate table/column names before interpolating into SQL.
        """
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier!r}")
        return identifier

    def fetch_docs_data(self):
        """
        Fetches chunk text and metadata from the specified table.
        Falls back to text-only when METADATA is unavailable.
        """
        _results = []
        table_name = self._validate_identifier(self.table_name)
        text_column = self._validate_identifier(self.text_column)

        with get_connection() as conn:
            with conn.cursor() as cursor:
                query_with_metadata = f"""
                    SELECT
                        {text_column},
                        json_value(METADATA, '$.source') AS source,
                        json_value(METADATA, '$.page_label') AS page_label
                    FROM {table_name}
                """

                query_text_only = f"SELECT {text_column} FROM {table_name}"

                has_metadata = True
                try:
                    cursor.execute(query_with_metadata)
                except oracledb.DatabaseError:
                    has_metadata = False
                    cursor.execute(query_text_only)

                while True:
                    # Fetch records in batches
                    rows = cursor.fetchmany(self.batch_size)
                    if not rows:
                        # Exit loop when no more data
                        break

                    for row in rows:
                        # This is a CLOB object
                        lob_data = row[0]

                        if isinstance(lob_data, oracledb.LOB):
                            # Read LOB content
                            chunk_text = lob_data.read()
                        else:
                            # Fallback for non-LOB data
                            chunk_text = str(lob_data)

                        if has_metadata:
                            source = row[1] or table_name
                            page_label = row[2] or ""
                        else:
                            source = table_name
                            page_label = ""

                        _results.append(
                            {
                                "page_content": chunk_text,
                                "metadata": {
                                    "source": source,
                                    "page_label": page_label,
                                },
                            }
                        )

        return _results

    def simple_tokenize(self, text):
        """
        Tokenizes a string by extracting words (alphanumeric sequences).

        :param text: Input text string
        :return: List of lowercase tokens
        """
        return re.findall(r"\w+", text.lower())

    def index_data(self):
        """
        Reads text from the database and prepares BM25 index.
        """
        logger.info("Creating BM25 index...")
        try:
            self.docs = self.fetch_docs_data()
        except oracledb.DatabaseError as exc:
            logger.error("Failed to fetch data for BM25 indexing: %s", exc)
            self.docs = []
            self.texts = []
            self.tokenized_texts = []
            self.bm25 = None
            return

        valid_docs = [doc for doc in self.docs if doc.get("page_content")]
        self.docs = valid_docs
        self.texts = [doc["page_content"] for doc in self.docs]
        self.tokenized_texts = [self.simple_tokenize(text) for text in self.texts]
        if not self.tokenized_texts:
            logger.warning("No text available for BM25 index creation.")
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(self.tokenized_texts)

        logger.info("BM25 index created successfully!")
        logger.info("")

    def search(self, query, top_n=5):
        """
        Performs a BM25 search on the indexed documents.

        :param query: Search query string
        :param top_n: Number of top results to return
        :return: List of tuples (text, score)
        """
        if not self.bm25:
            logger.warning("BM25 index not initialized. Please check data indexing.")
            return []

        if not query or not query.strip():
            logger.warning("Empty query received in BM25 search.")
            return []

        if top_n <= 0:
            return []

        query_tokens = self.simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_n]  # Get top_n results

        _results = [(self.texts[i], scores[i]) for i in ranked_indices]

        return _results

    def search_docs(self, query, top_n=5):
        """
        Performs a BM25 search returning full docs with metadata.
        """
        if not self.bm25:
            logger.warning("BM25 index not initialized. Please check data indexing.")
            return []

        if not query or not query.strip():
            logger.warning("Empty query received in BM25 search.")
            return []

        if top_n <= 0:
            return []

        query_tokens = self.simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_n]

        return [
            {
                "page_content": self.docs[i]["page_content"],
                "metadata": self.docs[i]["metadata"] or {},
                "score": float(scores[i]),
            }
            for i in ranked_indices
        ]


# Example usage: DB credentials are resolved by `core.db_utils.get_connection()`.


def run_test():
    """
    To run a quick test.
    """
    table_name = "COLL01"
    text_column = "TEXT"

    # create the index
    bm25_search = BM25OracleSearch(table_name, text_column)

    questions = [
        "Qual è il valore di resistenza elettrica per considerare un pavimento come statico dissipativo?",
    ]

    for _question in questions:
        results = bm25_search.search(_question, top_n=2)

        # Print search results
        for text, score in results:
            print(f"Score: {score:.2f} - Text:\n{text}")
            print("")


if __name__ == "__main__":
    run_test()
