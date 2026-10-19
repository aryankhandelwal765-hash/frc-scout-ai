from sqlalchemy.orm import Session

from app.models import AuditLog


def audit(db: Session, user_id: int | None, action: str, detail: str | None = None) -> None:
    """Add an audit row to the session. The caller commits."""
    db.add(AuditLog(user_id=user_id, action=action, detail=detail))
