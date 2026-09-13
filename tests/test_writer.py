"""Tests for the writer module."""

import csv
import os
import sys
import tempfile
import unittest
from decimal import Decimal

# Add code directory to path
code_dir = os.path.join(os.path.dirname(__file__), "..", "code")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Also add shared checkout code directory for git worktree environments
shared_code_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "code"
)
if os.path.exists(shared_code_dir) and shared_code_dir not in sys.path:
    sys.path.insert(0, shared_code_dir)

from contracts import Decision, OUTPUT_COLUMNS
from writer import format_amount_plain, decision_row, write_rows, fill_template


class TestFormatAmountPlain(unittest.TestCase):
    """Tests for format_amount_plain function."""

    def test_integral_no_decimals(self):
        """Test integral values without decimal point."""
        self.assertEqual(format_amount_plain(Decimal("25256")), "25256")
        self.assertEqual(format_amount_plain(Decimal("0")), "0")
        self.assertEqual(format_amount_plain(Decimal("1425000")), "1425000")

    def test_decimal_one_place(self):
        """Test decimal values with one place."""
        self.assertEqual(format_amount_plain(Decimal("603.3")), "603.3")
        self.assertEqual(format_amount_plain(Decimal("23.5")), "23.5")

    def test_decimal_two_places(self):
        """Test decimal values with two places."""
        self.assertEqual(format_amount_plain(Decimal("87170.56")), "87170.56")

    def test_decimal_with_trailing_zeros(self):
        """Test decimal values with trailing zeros stripped."""
        self.assertEqual(format_amount_plain(Decimal("620.40")), "620.4")
        self.assertEqual(format_amount_plain(Decimal("15952906.67")), "15952906.67")


class TestDecisionRow(unittest.TestCase):
    """Tests for decision_row function."""

    def test_decision_row_order(self):
        """Test that decision_row returns columns in OUTPUT_COLUMNS order."""
        decision = Decision(
            request_id="request_01",
            amount_safe_to_pay=Decimal("25256"),
            affordability_status="affordable_now",
            recommended_payment_method="full_payment",
            payment_plan="2024-03-03:25256",
            earliest_date_for_full_payment="2024-03-03",
            spending_changes_needed="none",
            decision_explanation="Pay ZAR 25,256 today.",
        )

        row = decision_row(decision)

        # Check that it's a list with the right number of columns
        self.assertEqual(len(row), len(OUTPUT_COLUMNS))

        # Check that values match columns
        self.assertEqual(row[0], "request_01")
        self.assertEqual(row[1], "25256")
        self.assertEqual(row[2], "affordable_now")
        self.assertEqual(row[3], "full_payment")
        self.assertEqual(row[4], "2024-03-03:25256")
        self.assertEqual(row[5], "2024-03-03")
        self.assertEqual(row[6], "none")
        self.assertEqual(row[7], "Pay ZAR 25,256 today.")

    def test_decision_row_with_decimal_formatting(self):
        """Test that decision_row uses format_amount_plain for amount_safe_to_pay."""
        decision = Decision(
            request_id="request_02",
            amount_safe_to_pay=Decimal("620.40"),
            affordability_status="affordable_with_plan",
            recommended_payment_method="installments",
            payment_plan="2025-08-08:15952906.67|2025-09-07:15952906.67",
            earliest_date_for_full_payment="2025-09-15",
            spending_changes_needed="none",
            decision_explanation="Use 3 installments.",
        )

        row = decision_row(decision)

        # Check that amount is formatted without thousands separators and trailing zeros
        self.assertEqual(row[1], "620.4")


class TestWriteRows(unittest.TestCase):
    """Tests for write_rows function."""

    def test_write_rows_creates_csv(self):
        """Test that write_rows creates a valid CSV file."""
        decisions = [
            Decision(
                request_id="request_01",
                amount_safe_to_pay=Decimal("25256"),
                affordability_status="affordable_now",
                recommended_payment_method="full_payment",
                payment_plan="2024-03-03:25256",
                earliest_date_for_full_payment="2024-03-03",
                spending_changes_needed="none",
                decision_explanation="Pay ZAR 25,256 today.",
            ),
            Decision(
                request_id="request_02",
                amount_safe_to_pay=Decimal("620.4"),
                affordability_status="affordable_with_plan",
                recommended_payment_method="installments",
                payment_plan="2025-08-08:620.4",
                earliest_date_for_full_payment="2025-09-15",
                spending_changes_needed="none",
                decision_explanation="Use 3 installments.",
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            # Write rows
            row_count = write_rows(decisions, temp_path)

            # Check row count includes header
            self.assertEqual(row_count, 3)  # 1 header + 2 decisions

            # Read back and verify
            with open(temp_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["request_id"], "request_01")
            self.assertEqual(rows[0]["amount_safe_to_pay"], "25256")
            self.assertEqual(rows[1]["request_id"], "request_02")
            self.assertEqual(rows[1]["amount_safe_to_pay"], "620.4")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_write_rows_uses_utf8(self):
        """Test that write_rows uses UTF-8 encoding."""
        decision = Decision(
            request_id="request_01",
            amount_safe_to_pay=Decimal("25256"),
            affordability_status="affordable_now",
            recommended_payment_method="full_payment",
            payment_plan="2024-03-03:25256",
            earliest_date_for_full_payment="2024-03-03",
            spending_changes_needed="none",
            decision_explanation="Pay ZAR 25,256 today.",
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            write_rows([decision], temp_path)

            # Read as UTF-8 and verify it works
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("request_01", content)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestFillTemplate(unittest.TestCase):
    """Tests for fill_template function."""

    def test_fill_template_basic(self):
        """Test that fill_template fills template rows with decision data."""
        # Create a template file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv", newline=""
        ) as f:
            template_path = f.name
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerow({col: "" if col != "request_id" else "request_01" for col in OUTPUT_COLUMNS})
            writer.writerow({col: "" if col != "request_id" else "request_02" for col in OUTPUT_COLUMNS})

        # Create decisions
        decisions = [
            Decision(
                request_id="request_01",
                amount_safe_to_pay=Decimal("25256"),
                affordability_status="affordable_now",
                recommended_payment_method="full_payment",
                payment_plan="2024-03-03:25256",
                earliest_date_for_full_payment="2024-03-03",
                spending_changes_needed="none",
                decision_explanation="Pay ZAR 25,256 today.",
            ),
            Decision(
                request_id="request_02",
                amount_safe_to_pay=Decimal("620.4"),
                affordability_status="affordable_with_plan",
                recommended_payment_method="installments",
                payment_plan="2025-08-08:620.4",
                earliest_date_for_full_payment="2025-09-15",
                spending_changes_needed="none",
                decision_explanation="Use 3 installments.",
            ),
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as f:
            output_path = f.name

        try:
            # Fill template
            row_count = fill_template(decisions, template_path, output_path)

            # Check row count
            self.assertEqual(row_count, 3)  # 1 header + 2 decisions

            # Read output and verify
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["request_id"], "request_01")
            self.assertEqual(rows[0]["amount_safe_to_pay"], "25256")
            self.assertEqual(rows[0]["affordability_status"], "affordable_now")
            self.assertEqual(rows[1]["request_id"], "request_02")
            self.assertEqual(rows[1]["amount_safe_to_pay"], "620.4")
        finally:
            if os.path.exists(template_path):
                os.unlink(template_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_fill_template_missing_request_id(self):
        """Test that fill_template raises ValueError for missing request_id in decisions."""
        # Create a template file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv", newline=""
        ) as f:
            template_path = f.name
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerow({col: "" if col != "request_id" else "request_01" for col in OUTPUT_COLUMNS})
            writer.writerow({col: "" if col != "request_id" else "request_02" for col in OUTPUT_COLUMNS})

        # Create decisions with only one request_id
        decisions = [
            Decision(
                request_id="request_01",
                amount_safe_to_pay=Decimal("25256"),
                affordability_status="affordable_now",
                recommended_payment_method="full_payment",
                payment_plan="2024-03-03:25256",
                earliest_date_for_full_payment="2024-03-03",
                spending_changes_needed="none",
                decision_explanation="Pay ZAR 25,256 today.",
            ),
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as f:
            output_path = f.name

        try:
            # Should raise ValueError
            with self.assertRaises(ValueError):
                fill_template(decisions, template_path, output_path)
        finally:
            if os.path.exists(template_path):
                os.unlink(template_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_fill_template_extra_request_id(self):
        """Test that fill_template raises ValueError for extra request_id in decisions."""
        # Create a template file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv", newline=""
        ) as f:
            template_path = f.name
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerow({col: "" if col != "request_id" else "request_01" for col in OUTPUT_COLUMNS})

        # Create decisions with two request_ids
        decisions = [
            Decision(
                request_id="request_01",
                amount_safe_to_pay=Decimal("25256"),
                affordability_status="affordable_now",
                recommended_payment_method="full_payment",
                payment_plan="2024-03-03:25256",
                earliest_date_for_full_payment="2024-03-03",
                spending_changes_needed="none",
                decision_explanation="Pay ZAR 25,256 today.",
            ),
            Decision(
                request_id="request_02",
                amount_safe_to_pay=Decimal("620.4"),
                affordability_status="affordable_with_plan",
                recommended_payment_method="installments",
                payment_plan="2025-08-08:620.4",
                earliest_date_for_full_payment="2025-09-15",
                spending_changes_needed="none",
                decision_explanation="Use 3 installments.",
            ),
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as f:
            output_path = f.name

        try:
            # Should raise ValueError
            with self.assertRaises(ValueError):
                fill_template(decisions, template_path, output_path)
        finally:
            if os.path.exists(template_path):
                os.unlink(template_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_fill_template_preserves_original_on_error(self):
        """Test that fill_template doesn't modify original file on error."""
        # Create a template file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv", newline=""
        ) as f:
            template_path = f.name
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerow({col: "" if col != "request_id" else "request_01" for col in OUTPUT_COLUMNS})
            writer.writerow({col: "" if col != "request_id" else "request_02" for col in OUTPUT_COLUMNS})

        # Get original content
        with open(template_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Create decisions with only one request_id
        decisions = [
            Decision(
                request_id="request_01",
                amount_safe_to_pay=Decimal("25256"),
                affordability_status="affordable_now",
                recommended_payment_method="full_payment",
                payment_plan="2024-03-03:25256",
                earliest_date_for_full_payment="2024-03-03",
                spending_changes_needed="none",
                decision_explanation="Pay ZAR 25,256 today.",
            ),
        ]

        try:
            # This should raise ValueError
            with self.assertRaises(ValueError):
                fill_template(decisions, template_path)  # default: output_path = template_path

            # Verify original file is unchanged
            with open(template_path, "r", encoding="utf-8") as f:
                current_content = f.read()

            self.assertEqual(current_content, original_content)
        finally:
            if os.path.exists(template_path):
                os.unlink(template_path)


if __name__ == "__main__":
    unittest.main()
