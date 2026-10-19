"""initial schema (docs/BUILD_SPEC.md) + invite_codes.code_prefix

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

NOW = sa.text("(datetime('now'))")
ROLE_CK = "role IN ('scout','strategist','admin')"


def _id():
    return sa.Column("id", sa.Integer, primary_key=True, autoincrement=True)


def _created(name="created_at"):
    return sa.Column(name, sa.Text, nullable=False, server_default=NOW)


def upgrade() -> None:
    ai = {"sqlite_autoincrement": True}

    op.create_table(
        "users",
        _id(),
        sa.Column("username", sa.Text, nullable=False, unique=True),
        sa.Column("display_name", sa.Text),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("active", sa.Integer, nullable=False, server_default="1"),
        _created(),
        sa.CheckConstraint(ROLE_CK, name="ck_users_role"),
        sa.CheckConstraint("active IN (0,1)", name="ck_users_active"),
        **ai,
    )
    op.create_table(
        "invite_codes",
        _id(),
        sa.Column("code_hash", sa.Text, nullable=False),
        # Deviation from spec: lookup handle, since code_hash is salted argon2.
        sa.Column("code_prefix", sa.Text, unique=True),
        sa.Column("role", sa.Text, nullable=False, server_default="scout"),
        sa.Column("expires_at", sa.Text),
        sa.Column("used_by", sa.Integer, sa.ForeignKey("users.id")),
        sa.Column("created_by", sa.Integer, sa.ForeignKey("users.id")),
        _created(),
        sa.CheckConstraint(ROLE_CK, name="ck_invite_role"),
        **ai,
    )
    op.create_table(
        "app_settings",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("value", sa.Text, nullable=False),
    )
    op.create_table(
        "audit_log",
        _id(),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id")),
        sa.Column("action", sa.Text, nullable=False),
        sa.Column("detail", sa.Text),
        _created(),
        **ai,
    )
    op.create_table(
        "seasons",
        sa.Column("year", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("game_name", sa.Text),
        sa.Column("scouting_schema", sa.Text, nullable=False),
        sa.Column("prescout_schema", sa.Text),
    )
    op.create_table(
        "events",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("season", sa.Integer, sa.ForeignKey("seasons.year"), nullable=False),
        sa.Column("name", sa.Text),
        sa.Column("start_date", sa.Text),
        sa.Column("end_date", sa.Text),
    )
    op.create_table(
        "frc_teams",
        sa.Column("number", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("name", sa.Text),
        sa.Column("city", sa.Text),
        sa.Column("state_prov", sa.Text),
    )
    op.create_table(
        "matches",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("event_key", sa.Text, sa.ForeignKey("events.key"), nullable=False),
        sa.Column("comp_level", sa.Text),
        sa.Column("set_number", sa.Integer),
        sa.Column("match_number", sa.Integer),
        sa.Column("red_score", sa.Integer),
        sa.Column("blue_score", sa.Integer),
        sa.Column("played", sa.Integer, nullable=False, server_default="0"),
    )
    op.create_table(
        "match_teams",
        sa.Column("match_key", sa.Text, sa.ForeignKey("matches.key"), primary_key=True),
        sa.Column("team_number", sa.Integer, sa.ForeignKey("frc_teams.number"), primary_key=True),
        sa.Column("alliance", sa.Text, nullable=False),
        sa.Column("slot", sa.Integer),
        sa.CheckConstraint("alliance IN ('red','blue')", name="ck_match_teams_alliance"),
    )
    op.create_index("idx_match_teams_team", "match_teams", ["team_number"])
    op.create_table(
        "external_stats",
        _id(),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("team_number", sa.Integer),
        sa.Column("event_key", sa.Text),
        sa.Column("data", sa.Text, nullable=False),
        _created("fetched_at"),
        sa.CheckConstraint("source IN ('tba','statbotics')", name="ck_external_source"),
        **ai,
    )
    op.create_table(
        "scouting_entries",
        _id(),
        sa.Column("match_key", sa.Text, sa.ForeignKey("matches.key"), nullable=False),
        sa.Column("team_number", sa.Integer, sa.ForeignKey("frc_teams.number"), nullable=False),
        sa.Column("scouter_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("data", sa.Text, nullable=False),
        sa.Column("notes", sa.Text),
        sa.Column("status", sa.Text, nullable=False, server_default="ok"),
        sa.Column("client_uuid", sa.Text, unique=True),
        _created("submitted_at"),
        sa.CheckConstraint("status IN ('ok','flagged','rejected')", name="ck_scouting_status"),
        **ai,
    )
    op.create_index("idx_scout_team", "scouting_entries", ["team_number"])
    op.create_index("idx_scout_match", "scouting_entries", ["match_key"])
    op.create_table(
        "pit_scouting",
        _id(),
        sa.Column("event_key", sa.Text, sa.ForeignKey("events.key"), nullable=False),
        sa.Column("team_number", sa.Integer, sa.ForeignKey("frc_teams.number"), nullable=False),
        sa.Column("scouter_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("data", sa.Text, nullable=False),
        _created("submitted_at"),
        **ai,
    )
    op.create_table(
        "pit_photos",
        _id(),
        sa.Column(
            "pit_id", sa.Integer, sa.ForeignKey("pit_scouting.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("path", sa.Text, nullable=False),
        **ai,
    )
    op.create_table(
        "prescout_batches",
        _id(),
        sa.Column("target_event_key", sa.Text, sa.ForeignKey("events.key")),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("source_type", sa.Text, nullable=False),
        sa.Column("source_file", sa.Text),
        sa.Column("created_by", sa.Integer, sa.ForeignKey("users.id")),
        _created(),
        sa.CheckConstraint(
            "source_type IN ('csv','video','manual','api')", name="ck_prescout_source_type"
        ),
        **ai,
    )
    op.create_table(
        "prescout_entries",
        _id(),
        sa.Column(
            "batch_id", sa.Integer, sa.ForeignKey("prescout_batches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("team_number", sa.Integer, sa.ForeignKey("frc_teams.number"), nullable=False),
        sa.Column("source_event_key", sa.Text, sa.ForeignKey("events.key")),
        sa.Column("source_match_key", sa.Text, sa.ForeignKey("matches.key")),
        sa.Column("data", sa.Text, nullable=False),
        sa.Column("video_url", sa.Text),
        sa.Column("notes", sa.Text),
        sa.Column("confidence", sa.Integer),
        sa.Column("scouter_id", sa.Integer, sa.ForeignKey("users.id")),
        _created(),
        sa.CheckConstraint("confidence BETWEEN 1 AND 5", name="ck_prescout_confidence"),
        **ai,
    )
    op.create_index("idx_prescout_team", "prescout_entries", ["team_number"])
    op.create_index("idx_prescout_batch", "prescout_entries", ["batch_id"])
    op.create_table(
        "video_tracks",
        _id(),
        sa.Column("match_key", sa.Text, sa.ForeignKey("matches.key")),
        sa.Column("team_number", sa.Integer),
        sa.Column("data", sa.Text, nullable=False),
        _created(),
        **ai,
    )
    op.create_table(
        "ai_reports",
        _id(),
        sa.Column("event_key", sa.Text),
        sa.Column("team_number", sa.Integer),
        sa.Column("match_key", sa.Text),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("model", sa.Text),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("inputs_hash", sa.Text),
        _created(),
        **ai,
    )
    op.create_table(
        "picklists",
        _id(),
        sa.Column("event_key", sa.Text, sa.ForeignKey("events.key"), nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("created_by", sa.Integer, sa.ForeignKey("users.id")),
        _created(),
        **ai,
    )
    op.create_table(
        "picklist_items",
        sa.Column(
            "picklist_id", sa.Integer, sa.ForeignKey("picklists.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("team_number", sa.Integer, sa.ForeignKey("frc_teams.number"), primary_key=True),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column("tier", sa.Text),
        sa.Column("note", sa.Text),
    )


def downgrade() -> None:
    for t in (
        "picklist_items", "picklists", "ai_reports", "video_tracks", "prescout_entries",
        "prescout_batches", "pit_photos", "pit_scouting", "scouting_entries", "external_stats",
        "match_teams", "matches", "frc_teams", "events", "seasons", "audit_log",
        "app_settings", "invite_codes", "users",
    ):
        op.drop_table(t)
