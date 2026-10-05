from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from datetime import date as Date, datetime, time as Time, timedelta
from decimal import Decimal

# ==================== Offer Schemas ====================

# Mirrors DecimalField(max_digits=10, decimal_places=2) on the backend, minus negatives
Money = Annotated[Decimal, Field(ge=0, max_digits=10, decimal_places=2)]


class OfferDraft(BaseModel):
    """
    Contract fields of a GetAvails offer, as drafted by the generate_offer tool
    and returned to the client in `data` when `response_type` is "offer".

    Mirrors CONTRACT_FIELDS and the Offer model in the main backend
    (apps/offers): the same names, lengths and number formats, so a completed
    draft is accepted by POST /api/v1/offers/. Keep the two in sync. The app's
    offer form completes it: it fills in the signatory and buyer details the
    draft leaves null, and the user adds the recipient, signature and
    documents, which are not part of a draft.

    Field descriptions are what the model sees as the tool's argument docs.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    artist_name: str = Field(max_length=128, description="Name of the artist or act the offer is for")
    date: Date = Field(description="Event date as YYYY-MM-DD")
    venue: str = Field(max_length=128, description="Name of the venue")
    venue_address: str = Field(max_length=256, description="Street address of the venue")
    city_state_country_zip: str = Field(
        max_length=128,
        description="Venue city, state, country and postal code on one line, e.g. 'Austin, TX, USA, 78701'",
    )
    venue_phone: str = Field(max_length=32, description="Venue phone number")

    offer_amount: Money = Field(
        description="Amount offered to the artist, as a plain number such as 1500 or 1500.50 "
        "(no currency symbol or thousands separators)"
    )
    airfare: Optional[Money] = Field(None, description="Airfare amount, if part of the offer")
    backline: Optional[Money] = Field(None, description="Backline amount, if part of the offer")
    hotel_ground_transportation: Optional[Money] = Field(
        None, description="Hotel and ground transportation amount, if part of the offer"
    )
    catering: Optional[Money] = Field(None, description="Catering amount, if part of the offer")
    first_class_sound_and_lighting: Optional[Money] = Field(
        None, description="First-class sound and lighting amount, if part of the offer"
    )

    door_time: Time = Field(description="Door time as 24-hour HH:MM, e.g. 19:30")
    expected_attendance: int = Field(ge=1, le=2_147_483_647, description="Expected number of attendees")
    past_performers: Optional[str] = Field(
        None, max_length=256, description="Artists who have played this venue or event before"
    )
    social_media_request: Optional[str] = Field(
        None, max_length=256, description="Social media or promotion requested from the artist"
    )
    what_is_event_for: Optional[str] = Field(
        None, max_length=256, description="What the event is for, e.g. a festival, private party or fundraiser"
    )
    other_artists: Optional[str] = Field(None, max_length=256, description="Other artists on the bill")

    # Required by the backend but optional in a draft: the offer form fills the
    # signatory and buyer in from the user's profile, so Ava doesn't ask for them
    contact_signatory_name: Optional[str] = Field(
        None, max_length=128, description="Full name of the signatory, the person signing for the buyer"
    )
    contact_signatory_address: Optional[str] = Field(None, max_length=256, description="Signatory's address")
    contact_signatory_contact_info: Optional[str] = Field(
        None, max_length=32, description="Signatory's phone number or email (32 characters at most)"
    )
    contact_buyer_name: Optional[str] = Field(
        None, max_length=128, description="Name of the buyer making the offer"
    )
    contact_buyer_address: Optional[str] = Field(None, max_length=256, description="Buyer's address")
    contact_buyer_contact_info: Optional[str] = Field(
        None, max_length=32, description="Buyer's phone number or email (32 characters at most)"
    )
    contact_production_name: str = Field(max_length=128, description="Name of the production contact")
    contact_production_contact_info: str = Field(
        max_length=32, description="Production contact's phone number or email (32 characters at most)"
    )

    included_facilities: List[str] = Field(
        default_factory=list,
        description="Facilities included with the offer, each as a short label, e.g. ['Green room', 'Parking']",
    )
    additional_notes: Optional[str] = Field(None, description="Any other terms or notes for the offer")

    @model_validator(mode="before")
    @classmethod
    def _drop_blank_values(cls, data: Any) -> Any:
        """Treat null and blank values as not provided, so they read as missing rather than invalid."""
        if not isinstance(data, dict):
            return data
        return {
            key: value for key, value in data.items()
            if value is not None and not (isinstance(value, str) and not value.strip())
        }

    @field_validator("date", "door_time", mode="before")
    @classmethod
    def _reject_numbers(cls, value: Any) -> Any:
        # Pydantic would read a bare number as a Unix timestamp or seconds past midnight
        if isinstance(value, (int, float)):
            raise ValueError("use the text format given in the field description")
        return value

    @field_validator("date")
    @classmethod
    def _not_in_the_past(cls, value: Date) -> Date:
        # A day of slack: the user may be in a timezone behind the server's
        today = Date.today()
        if value < today - timedelta(days=1):
            raise ValueError(f"{value} is in the past (today is {today}); check the year with the user")
        return value

    @field_validator("door_time")
    @classmethod
    def _plain_time(cls, value: Time) -> Time:
        """Drop the timezone and fractions of a second, which the backend's TimeField rejects."""
        return value.replace(microsecond=0, tzinfo=None)

    @field_validator(
        "offer_amount", "airfare", "backline", "hotel_ground_transportation",
        "catering", "first_class_sound_and_lighting",
    )
    @classmethod
    def _two_decimal_places(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        """Serialize amounts the way the backend does: 1500 -> "1500.00"."""
        return None if value is None else value.quantize(Decimal("0.01"))

    @field_validator("included_facilities")
    @classmethod
    def _clean_facilities(cls, value: List[str]) -> List[str]:
        return [item for item in value if item]


# ==================== Chat Schemas ====================

# What an assistant reply carries. "message" is plain text only; the others
# mean `data` holds structured results for the client to render.
# Keep in sync with TOOL_RESPONSE_TYPES in src/tools.py.
ResponseType = Literal["message", "artists", "venues", "offer"]

# Structured payload: a list of results (artists/venues) or one object (offer)
ResponseData = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]

class ChatMessage(BaseModel):
    role: str
    content: str
    response_type: ResponseType = "message"
    data: ResponseData = None
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    content: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    response_type: ResponseType = "message"
    data: ResponseData = None
    session_id: str
    timestamp: datetime

class PublicChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    session_id: str
    timestamp: datetime

class ChatSession(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ChatHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    page: int
    limit: int
    total_messages: int
    total_pages: int

class SessionList(BaseModel):
    sessions: List[ChatSession]
    page: int
    limit: int
    total: int
    total_pages: int

class TitleUpdateRequest(BaseModel):
    session_id: str
    title: str