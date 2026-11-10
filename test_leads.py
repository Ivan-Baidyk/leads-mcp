"""Tests for the leads MCP server.

Run from the project root with the same interpreter that has `mcp` installed:

    python test_leads.py

The tests run against an isolated COPY of leads.json (set via LEADS_PATH), so
they never modify the real seed file.
"""

import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.join(HERE, "leads.json")

# Isolate: point the server at a throwaway copy before importing it.
_TMP = tempfile.mkdtemp(prefix="leads-test-")
os.environ["LEADS_PATH"] = os.path.join(_TMP, "leads.json")

sys.path.insert(0, HERE)
import leads_server as ls  # noqa: E402


class TestLeadsMcp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.copy2(SEED, os.environ["LEADS_PATH"])

    def test_01_list_all_returns_15(self):
        result = ls.list_leads()
        self.assertEqual(result["total"], 15)
        self.assertEqual(len(result["leads"]), 15)

    def test_02_list_new_returns_14(self):
        # lead 15 is already 'processed'
        result = ls.list_leads(status="new")
        self.assertEqual(result["total"], 14)
        self.assertEqual(len(result["leads"]), 14)

    def test_03_update_invalid_category_rejected(self):
        result = ls.update_lead(1, category="vip", priority="high")
        self.assertIn("error", result)
        self.assertIn("invalid category", result["error"])

    def test_04_update_invalid_priority_rejected(self):
        result = ls.update_lead(1, category="sale", priority="urgent")
        self.assertIn("error", result)

    def test_05_spam_draft_reply_null(self):
        result = ls.update_lead(3, category="spam", priority="low", draft_reply="hello")
        self.assertIsNone(result.get("draft_reply"))
        self.assertIn("warning", result)

    def test_06_update_unknown_id_errors(self):
        result = ls.update_lead(999, category="sale", priority="low")
        self.assertEqual(result, {"error": "Lead 999 not found"})

    def test_07_atomic_update_preserves_other_leads(self):
        # after the writes above, the file must still have 15 leads
        self.assertEqual(len(ls._read_leads()), 15)
        # identity fields are untouched
        lead1 = ls.get_lead(1)
        self.assertEqual(lead1["name"], "Ольга Смирнова")

    def test_08_broken_json_returns_error(self):
        # corrupt the file: every tool returns {"error": ...}, never a silent []
        with open(os.environ["LEADS_PATH"], "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertIn("error", ls.list_leads())
        self.assertIn("error", ls.get_lead(1))
        self.assertIn("error", ls.update_lead(1, category="sale", priority="low"))
        shutil.copy2(SEED, os.environ["LEADS_PATH"])


if __name__ == "__main__":
    unittest.main()
