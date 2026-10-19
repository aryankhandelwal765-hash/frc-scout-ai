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
