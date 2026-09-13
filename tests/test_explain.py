"""Tests for the explain module."""

import os
import sys
import unittest
from datetime import date
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

# Now import after sys.path is configured
from contracts import (
    CandidatePlan,
    Payment,
    Profile,
    RequestRecord,
    SpendingChange,
)
from explain import fmt_amount, fmt_date, fmt_money, explain


class TestFmtAmount(unittest.TestCase):
    """Tests for fmt_amount function."""

    def test_integral_small(self):
        """Test integral values with thousands separators."""
        self.assertEqual(fmt_amount(Decimal("68432")), "68,432")

    def test_integral_large(self):
        """Test large integral values."""
        self.assertEqual(fmt_amount(Decimal("1425000")), "1,425,000")

    def test_decimal_two_places(self):
        """Test decimal values with exactly 2 decimal places."""
        self.assertEqual(fmt_amount(Decimal("620.40")), "620.40")
        self.assertEqual(fmt_amount(Decimal("23.50")), "23.50")
        self.assertEqual(fmt_amount(Decimal("95194.67")), "95,194.67")

    def test_decimal_one_place(self):
        """Test decimal values that need padding to 2 places."""
        self.assertEqual(fmt_amount(Decimal("23.5")), "23.50")

    def test_zero(self):
        """Test zero value."""
        self.assertEqual(fmt_amount(Decimal("0")), "0")

    def test_large_amount_with_decimals(self):
        """Test large amount with decimal places."""
        self.assertEqual(fmt_amount(Decimal("15952906.67")), "15,952,906.67")


class TestFmtDate(unittest.TestCase):
    """Tests for fmt_date function."""

    def test_no_leading_zero(self):
        """Test date formatting without leading zero."""
        self.assertEqual(fmt_date(date(2025, 8, 8)), "8 August 2025")

    def test_double_digit_day(self):
        """Test date with double-digit day."""
        self.assertEqual(fmt_date(date(2019, 11, 15)), "15 November 2019")

    def test_january(self):
        """Test January date."""
        self.assertEqual(fmt_date(date(2026, 1, 3)), "3 January 2026")


class TestFmtMoney(unittest.TestCase):
    """Tests for fmt_money function."""

    def test_currency_amount(self):
        """Test currency and amount formatting."""
        self.assertEqual(fmt_money("INR", Decimal("274600")), "INR 274,600")
        self.assertEqual(fmt_money("EUR", Decimal("620.40")), "EUR 620.40")
        self.assertEqual(fmt_money("ZAR", Decimal("25256")), "ZAR 25,256")


class TestExplain(unittest.TestCase):
    """Tests for explain function using sample data."""

    def test_mold_1a_request_01(self):
        """Test mold 1a: affordable_now + full_payment."""
        # request_01: pay ZAR 25,256 today, minimum 18,000
        request = RequestRecord(
            request_id="request_01",
            user_id="user_01",
            request_date=date(2024, 3, 3),
            request_type="purchase",
            requested_amount=Decimal("25256"),
            desired_completion_date=date(2024, 3, 20),
            allows_partial_payment=True,
            request_text="Would paying for the laptop today leave enough for my regular expenses?",
        )

        plan = CandidatePlan(
            method="full_payment",
            payments=(Payment(when=date(2024, 3, 3), amount=Decimal("25256")),),
            spending_changes=(),
            payment_option_id=None,
            total_paid=Decimal("25256"),
            completes_request=True,
        )

        profile = Profile(
            user_id="user_01",
            home_currency="ZAR",
            current_available_balance=Decimal("58481.1"),
            minimum_balance_to_keep=Decimal("18000"),
            financial_priorities=("education", "debt_repayment"),
            expense_categories_to_protect=("rent", "education", "groceries", "debt_repayment"),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("delivery_membership",),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )

        result = explain(request, plan, "affordable_now", profile)
        expected = "Pay ZAR 25,256 today. This leaves at least ZAR 18,000 available over the next 90 days."
        self.assertEqual(result, expected)

    def test_mold_2a_request_03(self):
        """Test mold 2a: affordable_later + wait."""
        # request_03: wait until 2019-11-15 to pay IDR 5,491,000
        request = RequestRecord(
            request_id="request_03",
            user_id="user_03",
            request_date=date(2019, 9, 3),
            request_type="education",
            requested_amount=Decimal("5491000"),
            desired_completion_date=date(2019, 11, 15),
            allows_partial_payment=False,
            request_text="Should I pay for the course now, use installments, or wait?",
        )

        plan = CandidatePlan(
            method="wait",
            payments=(Payment(when=date(2019, 11, 15), amount=Decimal("5491000")),),
            spending_changes=(),
            payment_option_id=None,
            total_paid=Decimal("5491000"),
            completes_request=True,
        )

        profile = Profile(
            user_id="user_03",
            home_currency="IDR",
            current_available_balance=Decimal("5810300"),
            minimum_balance_to_keep=Decimal("2668700"),
            financial_priorities=("retirement_investment", "emergency_savings"),
            expense_categories_to_protect=("rent", "utilities", "groceries"),
            expense_categories_user_is_willing_to_reduce=("streaming", "shopping"),
            expense_categories_user_is_willing_to_stop=("streaming", "cloud_storage"),
            payment_methods_user_will_consider=("full_payment", "partial_payment", "installments"),
            max_installment_months=2,
        )

        result = explain(request, plan, "affordable_later", profile)
        expected = "Pay IDR 5,491,000 in full on 15 November 2019. Paying earlier would take the balance below the IDR 2,668,700 minimum."
        self.assertEqual(result, expected)

    def test_mold_3_request_02(self):
        """Test mold 3: affordable_with_plan + installments."""
        # request_02: 3 installments of IDR 15,952,906.67
        request = RequestRecord(
            request_id="request_02",
            user_id="user_02",
            request_date=date(2025, 8, 5),
            request_type="travel",
            requested_amount=Decimal("46018000"),
            desired_completion_date=date(2025, 10, 10),
            allows_partial_payment=False,
            request_text="The current quote for the trip is IDR 46,018,000.",
        )

        plan = CandidatePlan(
            method="installments",
            payments=(
                Payment(when=date(2025, 8, 8), amount=Decimal("15952906.67")),
                Payment(when=date(2025, 9, 7), amount=Decimal("15952906.67")),
                Payment(when=date(2025, 10, 7), amount=Decimal("15952906.67")),
            ),
            spending_changes=(),
            payment_option_id=None,
            total_paid=Decimal("46018000"),
            completes_request=True,
        )

        profile = Profile(
            user_id="user_02",
            home_currency="IDR",
            current_available_balance=Decimal("60383889.2"),
            minimum_balance_to_keep=Decimal("29158400"),
            financial_priorities=("education", "family_support"),
            expense_categories_to_protect=("housing", "utilities", "education"),
            expense_categories_user_is_willing_to_reduce=("entertainment",),
            expense_categories_user_is_willing_to_stop=("cloud_storage",),
            payment_methods_user_will_consider=("partial_payment", "installments"),
            max_installment_months=7,
        )

        result = explain(request, plan, "affordable_with_plan", profile)
        expected = "Use 3 installments of IDR 15,952,906.67, starting 8 August 2025. This leaves at least IDR 29,158,400 available."
        self.assertEqual(result, expected)

    def test_mold_5_stop_request_06(self):
        """Test mold 5 with stop: stop family streaming plan."""
        # request_06: stop event_476, pay EUR 620.40
        request = RequestRecord(
            request_id="request_06",
            user_id="user_06",
            request_date=date(2026, 1, 3),
            request_type="investment",
            requested_amount=Decimal("620.4"),
            desired_completion_date=date(2026, 1, 14),
            allows_partial_payment=False,
            request_text="I want to put EUR 620.40 into an investment.",
        )

        plan = CandidatePlan(
            method="full_payment",
            payments=(Payment(when=date(2026, 1, 3), amount=Decimal("620.40")),),
            spending_changes=(SpendingChange(action="stop", event_id="event_476", new_amount=None),),
            payment_option_id=None,
            total_paid=Decimal("620.40"),
            completes_request=True,
        )

        profile = Profile(
            user_id="user_06",
            home_currency="EUR",
            current_available_balance=Decimal("1942.4"),
            minimum_balance_to_keep=Decimal("800"),
            financial_priorities=("emergency_savings", "travel"),
            expense_categories_to_protect=("rent", "insurance", "transport"),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=("streaming",),
            payment_methods_user_will_consider=("full_payment", "partial_payment"),
            max_installment_months=None,
        )

        descriptions = {"event_476": "Family streaming plan"}
        result = explain(request, plan, "affordable_with_plan", profile, descriptions=descriptions)
        expected = "Stop the family streaming plan, then pay EUR 620.40 today. This leaves at least EUR 800 available."
        self.assertEqual(result, expected)

    def test_mold_5_reduce_request_11(self):
        """Test mold 5 with reduce: reduce weekend food delivery."""
        # request_11: reduce event_989 to IDR 665,950
        request = RequestRecord(
            request_id="request_11",
            user_id="user_11",
            request_date=date(2025, 5, 3),
            request_type="travel",
            requested_amount=Decimal("13110000"),
            desired_completion_date=date(2025, 6, 12),
            allows_partial_payment=False,
            request_text="Does paying for the trip now leave enough for the rest of the month?",
        )

        plan = CandidatePlan(
            method="full_payment",
            payments=(Payment(when=date(2025, 5, 3), amount=Decimal("13110000")),),
            spending_changes=(
                SpendingChange(action="reduce_to", event_id="event_989", new_amount=Decimal("665950")),
            ),
            payment_option_id=None,
            total_paid=Decimal("13110000"),
            completes_request=True,
        )

        profile = Profile(
            user_id="user_11",
            home_currency="IDR",
            current_available_balance=Decimal("63531795"),
            minimum_balance_to_keep=Decimal("34140600"),
            financial_priorities=("education", "family_support"),
            expense_categories_to_protect=("housing", "utilities", "education"),
            expense_categories_user_is_willing_to_reduce=("dining", "entertainment"),
            expense_categories_user_is_willing_to_stop=("cloud_storage",),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )

        descriptions = {"event_989": "Weekend food delivery"}
        result = explain(request, plan, "affordable_with_plan", profile, descriptions=descriptions)
        expected = "Reduce the weekend food delivery to IDR 665,950, then pay IDR 13,110,000 today. This leaves at least IDR 34,140,600 available."
        self.assertEqual(result, expected)

    def test_mold_5_both_request_21(self):
        """Test mold 5 with both stop and reduce."""
        # request_21: stop event_1815 and reduce event_1816 to USD 23.50
        request = RequestRecord(
            request_id="request_21",
            user_id="user_21",
            request_date=date(2026, 4, 3),
            request_type="education",
            requested_amount=Decimal("1574.4"),
            desired_completion_date=date(2026, 4, 14),
            allows_partial_payment=False,
            request_text="Is it safe to cover the full course fee by the deadline?",
        )

        plan = CandidatePlan(
            method="full_payment",
            payments=(Payment(when=date(2026, 4, 3), amount=Decimal("1574.40")),),
            spending_changes=(
                SpendingChange(action="stop", event_id="event_1815", new_amount=None),
                SpendingChange(action="reduce_to", event_id="event_1816", new_amount=Decimal("23.50")),
            ),
            payment_option_id=None,
            total_paid=Decimal("1574.40"),
            completes_request=True,
        )

        profile = Profile(
            user_id="user_21",
            home_currency="USD",
            current_available_balance=Decimal("3911.35"),
            minimum_balance_to_keep=Decimal("1800"),
            financial_priorities=("retirement_investment", "emergency_savings"),
            expense_categories_to_protect=("rent", "utilities", "groceries"),
            expense_categories_user_is_willing_to_reduce=("dining", "streaming", "shopping"),
            expense_categories_user_is_willing_to_stop=("streaming", "cloud_storage"),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )

        descriptions = {
            "event_1815": "Online backup subscription",
            "event_1816": "Streaming subscription",
        }
        result = explain(request, plan, "affordable_with_plan", profile, descriptions=descriptions)
        expected = "Stop the online backup subscription and reduce the streaming subscription to USD 23.50, then pay USD 1,574.40 today. This leaves at least USD 1,800 available."
        self.assertEqual(result, expected)

    def test_mold_6_request_19(self):
        """Test mold 6: partial_payment."""
        # request_19: pay INR 28,820 today and INR 10,840 on 2024-09-15
        request = RequestRecord(
            request_id="request_19",
            user_id="user_19",
            request_date=date(2024, 9, 4),
            request_type="purchase",
            requested_amount=Decimal("39660"),
            desired_completion_date=date(2024, 10, 4),
            allows_partial_payment=True,
            request_text="Can I buy the laptop now without making next month's bills tight?",
        )

        plan = CandidatePlan(
            method="partial_payment",
            payments=(
                Payment(when=date(2024, 9, 4), amount=Decimal("28820")),
                Payment(when=date(2024, 9, 15), amount=Decimal("10840")),
            ),
            spending_changes=(),
            payment_option_id=None,
            total_paid=Decimal("39660"),
            completes_request=True,
        )

        profile = Profile(
            user_id="user_19",
            home_currency="INR",
            current_available_balance=Decimal("199545"),
            minimum_balance_to_keep=Decimal("92800"),
            financial_priorities=("healthcare", "family_support"),
            expense_categories_to_protect=("rent", "healthcare", "family_support", "groceries"),
            expense_categories_user_is_willing_to_reduce=("shopping",),
            expense_categories_user_is_willing_to_stop=("cloud_storage",),
            payment_methods_user_will_consider=("partial_payment", "installments"),
            max_installment_months=2,
        )

        result = explain(request, plan, "affordable_with_plan", profile)
        expected = "Pay INR 28,820 today and the remaining INR 10,840 on 15 September 2024. This completes the full request and keeps the INR 92,800 minimum protected."
        self.assertEqual(result, expected)

    def test_mold_4a_request_05(self):
        """Test mold 4a: not_affordable."""
        # request_05: do not proceed
        request = RequestRecord(
            request_id="request_05",
            user_id="user_05",
            request_date=date(2025, 11, 6),
            request_type="debt_repayment",
            requested_amount=Decimal("15488"),
            desired_completion_date=date(2026, 1, 12),
            allows_partial_payment=False,
            request_text="Can I clear this additional amount without putting upcoming bills at risk?",
        )

        plan = CandidatePlan(
            method="not_recommended",
            payments=(),
            spending_changes=(),
            payment_option_id=None,
            total_paid=Decimal("0"),
            completes_request=False,
        )

        profile = Profile(
            user_id="user_05",
            home_currency="ZAR",
            current_available_balance=Decimal("13837"),
            minimum_balance_to_keep=Decimal("13100"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=(),
            max_installment_months=None,
        )

        result = explain(request, plan, "not_affordable", profile)
        expected = "Do not make this payment by 12 January 2026. None of the available options keeps the ZAR 13,100 minimum protected."
        self.assertEqual(result, expected)

    def test_mold_4b_request_14(self):
        """Test mold 4b: not_affordable but some amount safe."""
        # request_14: amount_safe_to_pay is EUR 597.74, which is >= 10% of EUR 5,414.20
        request = RequestRecord(
            request_id="request_14",
            user_id="user_14",
            request_date=date(2025, 8, 4),
            request_type="debt_repayment",
            requested_amount=Decimal("5414.2"),
            desired_completion_date=date(2025, 10, 4),
            allows_partial_payment=True,
            request_text="I'm planning an extra loan payment of EUR 5,414.20.",
        )

        plan = CandidatePlan(
            method="not_recommended",
            payments=(),
            spending_changes=(),
            payment_option_id=None,
            total_paid=Decimal("0"),
            completes_request=False,
        )

        profile = Profile(
            user_id="user_14",
            home_currency="EUR",
            current_available_balance=Decimal("1000"),
            minimum_balance_to_keep=Decimal("400"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=(),
            max_installment_months=None,
        )

        result = explain(
            request,
            plan,
            "not_affordable",
            profile,
            amount_safe_to_pay=Decimal("597.74"),
        )
        expected = "Do not proceed with the EUR 5,414.20 request. Although EUR 597.74 is available today, the full amount cannot be completed safely within 90 days."
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
