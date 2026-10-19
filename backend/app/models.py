"""ORM models mirroring docs/BUILD_SPEC.md (plus invite_codes.code_prefix, see README)."""

from __future__ import annotations

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NOW = text("(datetime('now'))")
ROLES = ("scout", "strategist", "admin")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('scout','strategist','admin')", name="ck_users_role"),
        CheckConstraint("active IN (0,1)", name="ck_users_active"),
        {"sqlite_autoincrement": True},
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(Text, unique=True)
    display_name: Mapped[str | None] = mapped_column(Text)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text)
    active: Mapped[int] = mapped_column(Integer, server_default="1")
    created_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class InviteCode(Base):
    __tablename__ = "invite_codes"
    __table_args__ = (
        CheckConstraint("role IN ('scout','strategist','admin')", name="ck_invite_role"),
        {"sqlite_autoincrement": True},
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code_hash: Mapped[str] = mapped_column(Text)
    code_prefix: Mapped[str | None] = mapped_column(Text, unique=True)
    role: Mapped[str] = mapped_column(Text, server_default="scout")
    expires_at: Mapped[str | None] = mapped_column(Text)
    used_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class AppSetting(Base):
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = ({"sqlite_autoincrement": True},)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(Text)
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class Season(Base):
    __tablename__ = "seasons"
    year: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    game_name: Mapped[str | None] = mapped_column(Text)
    scouting_schema: Mapped[str] = mapped_column(Text)
    prescout_schema: Mapped[str | None] = mapped_column(Text)


class Event(Base):
    __tablename__ = "events"
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    season: Mapped[int] = mapped_column(ForeignKey("seasons.year"))
    name: Mapped[str | None] = mapped_column(Text)
    start_date: Mapped[str | None] = mapped_column(Text)
    end_date: Mapped[str | None] = mapped_column(Text)


class FrcTeam(Base):
    __tablename__ = "frc_teams"
    number: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(Text)
    state_prov: Mapped[str | None] = mapped_column(Text)


class Match(Base):
    __tablename__ = "matches"
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    event_key: Mapped[str] = mapped_column(ForeignKey("events.key"))
    comp_level: Mapped[str | None] = mapped_column(Text)
    set_number: Mapped[int | None] = mapped_column(Integer)
    match_number: Mapped[int | None] = mapped_column(Integer)
    red_score: Mapped[int | None] = mapped_column(Integer)
    blue_score: Mapped[int | None] = mapped_column(Integer)
    played: Mapped[int] = mapped_column(Integer, server_default="0")


class MatchTeam(Base):
    __tablename__ = "match_teams"
    __table_args__ = (
        CheckConstraint("alliance IN ('red','blue')", name="ck_match_teams_alliance"),
        Index("idx_match_teams_team", "team_number"),
    )
    match_key: Mapped[str] = mapped_column(ForeignKey("matches.key"), primary_key=True)
    team_number: Mapped[int] = mapped_column(ForeignKey("frc_teams.number"), primary_key=True)
    alliance: Mapped[str] = mapped_column(Text)
    slot: Mapped[int | None] = mapped_column(Integer)


class ExternalStat(Base):
    __tablename__ = "external_stats"
    __table_args__ = (
        CheckConstraint("source IN ('tba','statbotics')", name="ck_external_source"),
        {"sqlite_autoincrement": True},
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(Text)
    team_number: Mapped[int | None] = mapped_column(Integer)
    event_key: Mapped[str | None] = mapped_column(Text)
    data: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class ScoutingEntry(Base):
    __tablename__ = "scouting_entries"
    __table_args__ = (
        CheckConstraint("status IN ('ok','flagged','rejected')", name="ck_scouting_status"),
        Index("idx_scout_team", "team_number"),
        Index("idx_scout_match", "match_key"),
        {"sqlite_autoincrement": True},
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_key: Mapped[str] = mapped_column(ForeignKey("matches.key"))
    team_number: Mapped[int] = mapped_column(ForeignKey("frc_teams.number"))
    scouter_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    data: Mapped[str] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, server_default="ok")
    client_uuid: Mapped[str | None] = mapped_column(Text, unique=True)
    submitted_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class PitScouting(Base):
    __tablename__ = "pit_scouting"
    __table_args__ = ({"sqlite_autoincrement": True},)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_key: Mapped[str] = mapped_column(ForeignKey("events.key"))
    team_number: Mapped[int] = mapped_column(ForeignKey("frc_teams.number"))
    scouter_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    data: Mapped[str] = mapped_column(Text)
    submitted_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class PitPhoto(Base):
    __tablename__ = "pit_photos"
    __table_args__ = ({"sqlite_autoincrement": True},)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pit_id: Mapped[int] = mapped_column(ForeignKey("pit_scouting.id", ondelete="CASCADE"))
    path: Mapped[str] = mapped_column(Text)


class PrescoutBatch(Base):
    __tablename__ = "prescout_batches"
    __table_args__ = (
        CheckConstraint(
            "source_type IN ('csv','video','manual','api')", name="ck_prescout_source_type"
        ),
        {"sqlite_autoincrement": True},
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_event_key: Mapped[str | None] = mapped_column(ForeignKey("events.key"))
    name: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(Text)
    source_file: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class PrescoutEntry(Base):
    __tablename__ = "prescout_entries"
    __table_args__ = (
        CheckConstraint("confidence BETWEEN 1 AND 5", name="ck_prescout_confidence"),
        Index("idx_prescout_team", "team_number"),
        Index("idx_prescout_batch", "batch_id"),
        {"sqlite_autoincrement": True},
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("prescout_batches.id", ondelete="CASCADE"))
    team_number: Mapped[int] = mapped_column(ForeignKey("frc_teams.number"))
    source_event_key: Mapped[str | None] = mapped_column(ForeignKey("events.key"))
    source_match_key: Mapped[str | None] = mapped_column(ForeignKey("matches.key"))
    data: Mapped[str] = mapped_column(Text)
    video_url: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[int | None] = mapped_column(Integer)
    scouter_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class VideoTrack(Base):
    __tablename__ = "video_tracks"
    __table_args__ = ({"sqlite_autoincrement": True},)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_key: Mapped[str | None] = mapped_column(ForeignKey("matches.key"))
    team_number: Mapped[int | None] = mapped_column(Integer)
    data: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class AiReport(Base):
    __tablename__ = "ai_reports"
    __table_args__ = ({"sqlite_autoincrement": True},)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_key: Mapped[str | None] = mapped_column(Text)
    team_number: Mapped[int | None] = mapped_column(Integer)
    match_key: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    inputs_hash: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class Picklist(Base):
    __tablename__ = "picklists"
    __table_args__ = ({"sqlite_autoincrement": True},)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_key: Mapped[str] = mapped_column(ForeignKey("events.key"))
    name: Mapped[str] = mapped_column(Text)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[str] = mapped_column(Text, server_default=NOW)


class PicklistItem(Base):
    __tablename__ = "picklist_items"
    picklist_id: Mapped[int] = mapped_column(
        ForeignKey("picklists.id", ondelete="CASCADE"), primary_key=True
    )
    team_number: Mapped[int] = mapped_column(ForeignKey("frc_teams.number"), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    tier: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
