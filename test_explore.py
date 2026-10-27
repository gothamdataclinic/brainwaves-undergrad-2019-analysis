"""Run with: .venv/bin/python -m unittest test_explore.py"""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from explore import load_table


class LoadTableTest(unittest.TestCase):
    def test_qualtrics_records_scores_and_identifiers(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "survey.csv"
            rows = [
                ["Q9", "Q14_1", "Q14_2", "Q20", "Q20"],
                ["Student ID", "Question\nwith a newline", "Other question", "First", "Second"],
                ['{"ImportId":"id"}', '{"ImportId":"a"}', '{"ImportId":"b"}',
                 '{"ImportId":"c"}', '{"ImportId":"d"}'],
                ["001", "1 (Low)", "3", "yes", "no"],
                ["002", "5 (High)", "", "no", "yes"],
                ["003", "", "4", "yes", "yes"],
            ]
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            original = path.read_bytes()
            frame, labels = load_table(path)
            self.assertEqual(frame["Q9"].tolist(), ["001", "002", "003"])
            self.assertEqual(frame["Q14_1"].dropna().tolist(), [1, 5])
            self.assertTrue(pd.isna(frame.loc[2, "Q14_1"]))
            self.assertEqual(frame["Q14_2"].dropna().tolist(), [3, 4])
            self.assertEqual(frame[["Q20", "Q20.1"]].iloc[0].tolist(), ["yes", "no"])
            self.assertEqual(labels["Q14_1"], "Question\nwith a newline")
            self.assertEqual(path.read_bytes(), original)

            rows[-1][1] = "unexpected score"
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            with self.assertRaises(ValueError):
                load_table(path)

    def test_excel_missing_scores_are_not_zero_or_text(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scores.xlsx"
            pd.DataFrame({
                "ID": ["001", "002", "003"],
                "Pre": [0, "-", 4],
                "Post": [1, 2, "-"],
                "note": ["-", "keep", "keep"],
            }).to_excel(path, index=False)
            frame, _ = load_table(path)
            self.assertEqual(frame["Pre"].dropna().tolist(), [0, 4])
            self.assertEqual(frame["Post"].dropna().tolist(), [1, 2])
            self.assertEqual(frame[["Pre", "Post"]].dropna().values.tolist(), [[0, 1]])
            self.assertEqual(frame.loc[0, "note"], "-")


if __name__ == "__main__":
    unittest.main()
