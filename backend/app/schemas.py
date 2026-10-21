"""Explicit request/response models: these define the OpenAPI contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["scout", "strategist", "admin"]
USERNAME_PATTERN = r"^[A-Za-z0-9_.-]{3,32}$"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    display_name: str | None
    role: Role
    active: bool
    created_at: str


class OkOut(BaseModel):
    ok: bool = True


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class SignupIn(BaseModel):
    invite_code: str = Field(min_length=8, max_length=128)
    username: str = Field(pattern=USERNAME_PATTERN)
    password: str = Field(min_length=10, max_length=128)
    display_name: str | None = Field(default=None, max_length=64)


class InviteCreateIn(BaseModel):
    role: Role = "scout"
    expires_in_hours: int | None = Field(default=None, ge=1, le=24 * 365)


class InviteCreatedOut(BaseModel):
    """`code` is shown exactly once; only its argon2 hash is stored."""

    id: int
    code: str
    role: Role
    expires_at: str | None


class ImportSummaryOut(BaseModel):
    event_key: str
    season: int
    teams: int
    matches: int
    match_teams: int


class ErrorOut(BaseModel):
    detail: str


# --- events, scouting, pit, picklists, pre-scout, vision -----------------------------------------

EntryStatus = Literal["ok", "flagged", "rejected"]


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    key: str
    season: int
    name: str | None
    start_date: str | None
    end_date: str | None


class MatchTeamOut(BaseModel):
    team_number: int
    alliance: Literal["red", "blue"]
    slot: int | None


class MatchOut(BaseModel):
    key: str
    comp_level: str | None
    set_number: int | None
    match_number: int | None
    red_score: int | None
    blue_score: int | None
    played: bool
    teams: list[MatchTeamOut]


class SeasonOut(BaseModel):
    year: int
    game_name: str | None
    scouting_schema: dict
    prescout_schema: dict | None


class ScoutingEntryIn(BaseModel):
    match_key: str = Field(min_length=1, max_length=64)
    team_number: int = Field(ge=1, le=99999)
    data: dict
    notes: str | None = Field(default=None, max_length=2000)
    client_uuid: str | None = Field(default=None, min_length=8, max_length=64)


class ScoutingEntryOut(BaseModel):
    id: int
    match_key: str
    team_number: int
    scouter_id: int
    data: dict
    notes: str | None
    status: EntryStatus
    client_uuid: str | None
    submitted_at: str


class ScoutingBatchIn(BaseModel):
    entries: list[ScoutingEntryIn] = Field(min_length=1, max_length=200)


class ScoutingBatchItemOut(BaseModel):
    index: int
    client_uuid: str | None
    result: Literal["created", "duplicate", "invalid"]
    id: int | None = None
    errors: list[str] = []


class ScoutingBatchOut(BaseModel):
    items: list[ScoutingBatchItemOut]


class EntryStatusIn(BaseModel):
    status: EntryStatus


class PitIn(BaseModel):
    event_key: str = Field(min_length=1, max_length=64)
    team_number: int = Field(ge=1, le=99999)
    data: dict


class PitOut(BaseModel):
    id: int
    event_key: str
    team_number: int
    scouter_id: int
    data: dict
    submitted_at: str


class PicklistCreateIn(BaseModel):
    event_key: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=100)


class PicklistItemIO(BaseModel):
    team_number: int = Field(ge=1, le=99999)
    rank: int = Field(ge=1, le=1000)
    tier: str | None = Field(default=None, max_length=32)
    note: str | None = Field(default=None, max_length=500)


class PicklistOut(BaseModel):
    id: int
    event_key: str
    name: str
    created_by: int | None
    created_at: str
    items: list[PicklistItemIO] = []


class PicklistItemsIn(BaseModel):
    items: list[PicklistItemIO] = Field(max_length=200)


class PrescoutBatchIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source_type: Literal["csv", "video", "manual", "api"] = "manual"
    target_event_key: str | None = Field(default=None, max_length=64)


class PrescoutBatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    source_type: str
    target_event_key: str | None
    created_by: int | None
    created_at: str


class PrescoutEntryIn(BaseModel):
    team_number: int = Field(ge=1, le=99999)
    data: dict
    season_year: int = Field(ge=1992, le=2100)
    source_event_key: str | None = Field(default=None, max_length=64)
    source_match_key: str | None = Field(default=None, max_length=64)
    video_url: str | None = Field(default=None, max_length=500)
    notes: str | None = Field(default=None, max_length=2000)
    confidence: int | None = Field(default=None, ge=1, le=5)


class PrescoutEntryOut(BaseModel):
    id: int
    batch_id: int
    team_number: int
    source_event_key: str | None
    source_match_key: str | None
    data: dict
    video_url: str | None
    notes: str | None
    confidence: int | None
    scouter_id: int | None
    created_at: str


class VisionImportOut(BaseModel):
    event_key: str
    matches: int
    rows: int


class VisionRowOut(BaseModel):
    match_key: str
    team_number: int
    data: dict


class TeamSummaryOut(BaseModel):
    team_number: int
    name: str | None
    event_key: str
    matches_scouted: int
    scouting_means: dict[str, float]
    vision_matches: int
    vision_means: dict[str, float]
    prescout_entries: int
    pit_entries: int
