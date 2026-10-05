"""
Checks for the offer draft schema and the generate_offer tool.

Needs no API keys, database or network:

    python test/test_offer_draft.py
"""

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from src.schema import OfferDraft
from src.tools import generate_offer

EVENT_DATE = (date.today() + timedelta(days=90)).isoformat()


def offer_fields(**overrides) -> dict:
    """The required fields of an offer, as the model would send them."""
    fields = dict(
        artist_name="The Roadrunners",
        date=EVENT_DATE,
        venue="The Venue",
        venue_address="1 Main St",
        city_state_country_zip="Springfield, IL, USA, 00000",
        venue_phone="555-0100",
        offer_amount=1000,
        door_time="20:00",
        expected_attendance=200,
        contact_production_name="Prod Name",
        contact_production_contact_info="555-0103",
    )
    fields.update(overrides)
    return fields


def error_fields(**overrides) -> dict:
    """Validate an offer expected to fail; return {field: error type}."""
    try:
        OfferDraft(**offer_fields(**overrides))
    except ValidationError as e:
        return {str(err["loc"][0]): err["type"] for err in e.errors()}
    raise AssertionError("expected the offer to be rejected")


def call_tool(args: dict):
    return generate_offer.invoke(
        {"type": "tool_call", "name": "generate_offer", "args": args, "id": "call_1"}
    )


class OfferDraftTests(unittest.TestCase):
    def test_output_uses_the_backend_formats(self):
        offer = OfferDraft(**offer_fields(artist_name="  The Roadrunners ", airfare=250.5)).model_dump(mode="json")

        self.assertEqual(offer["artist_name"], "The Roadrunners")
        self.assertEqual(offer["date"], EVENT_DATE)
        self.assertEqual(offer["door_time"], "20:00:00")
        self.assertEqual(offer["offer_amount"], "1000.00")
        self.assertEqual(offer["airfare"], "250.50")
        # Unset optional fields are still present, as in the backend's responses
        self.assertIsNone(offer["catering"])
        self.assertEqual(offer["included_facilities"], [])
        self.assertEqual(len(offer), 28)

    def test_signatory_and_buyer_are_left_for_the_offer_form(self):
        offer = OfferDraft(**offer_fields())
        self.assertIsNone(offer.contact_signatory_name)
        self.assertIsNone(offer.contact_buyer_contact_info)
        # ...but kept when the user gave them in chat
        offer = OfferDraft(**offer_fields(contact_buyer_name="Buyer Name"))
        self.assertEqual(offer.contact_buyer_name, "Buyer Name")

    def test_null_and_blank_values_count_as_missing(self):
        self.assertEqual(
            error_fields(venue_phone=None, contact_production_name="   "),
            {"venue_phone": "missing", "contact_production_name": "missing"},
        )
        # ...and as unset for optional fields
        offer = OfferDraft(**offer_fields(catering=None, other_artists="", included_facilities=None))
        self.assertIsNone(offer.other_artists)
        self.assertEqual(offer.included_facilities, [])

    def test_backend_limits_are_enforced(self):
        self.assertIn("artist_name", error_fields(artist_name="x" * 129))
        self.assertIn("venue_address", error_fields(venue_address="x" * 257))
        self.assertIn("contact_buyer_contact_info", error_fields(contact_buyer_contact_info="x" * 33))
        self.assertIn("past_performers", error_fields(past_performers="x" * 257))
        # DecimalField(max_digits=10, decimal_places=2)
        self.assertIn("offer_amount", error_fields(offer_amount="1000.555"))
        self.assertIn("offer_amount", error_fields(offer_amount=100_000_000))
        OfferDraft(**offer_fields(offer_amount="99999999.99"))

    def test_values_that_make_no_sense_are_rejected(self):
        self.assertIn("offer_amount", error_fields(offer_amount=-1))
        self.assertIn("catering", error_fields(catering=-0.01))
        self.assertIn("offer_amount", error_fields(offer_amount="$1,000"))
        self.assertIn("expected_attendance", error_fields(expected_attendance=0))
        self.assertIn("date", error_fields(date="2020-06-01"))
        self.assertIn("date", error_fields(date="06/01/2030"))
        self.assertIn("door_time", error_fields(door_time="8pm"))
        # A bare number would otherwise be read as seconds past midnight
        self.assertIn("door_time", error_fields(door_time=20))

    def test_fields_outside_the_contract_are_rejected(self):
        self.assertEqual(error_fields(receiver_id=5), {"receiver_id": "extra_forbidden"})


class GenerateOfferToolTests(unittest.TestCase):
    def test_valid_offer_is_returned_as_the_artifact(self):
        message = call_tool(offer_fields())

        self.assertEqual(message.status, "success")
        self.assertEqual(message.artifact, OfferDraft(**offer_fields()).model_dump(mode="json"))

    def test_invalid_offer_tells_the_model_what_to_fix(self):
        fields = offer_fields(offer_amount="1000.555")
        del fields["venue_phone"]
        message = call_tool(fields)

        self.assertEqual(message.status, "error")
        # No artifact, so the reply carries no offer in `data`
        self.assertIsNone(message.artifact)
        self.assertIn("Missing required fields: venue_phone.", message.content)
        self.assertIn("- offer_amount:", message.content)


if __name__ == "__main__":
    unittest.main()
