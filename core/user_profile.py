"""
Resolve user profile from Oracle table USER_PROFILE.
"""

from core.db_utils import get_connection
from core.utils import get_console_logger

ALLOWED_PROFILES = {"ADMIN", "USER"}
DEFAULT_PROFILE = "USER"

logger = get_console_logger("user_profile")


def get_user_profile(username: str, default_profile: str = DEFAULT_PROFILE) -> str:
    """
    Return the effective profile for a username.
    Fallback to default profile on missing row/disabled user/errors.
    """
    _default = (default_profile or DEFAULT_PROFILE).strip().upper()
    if _default not in ALLOWED_PROFILES:
        _default = DEFAULT_PROFILE

    if not username:
        return _default

    sql = """
        SELECT profile_code, enabled
        FROM USER_PROFILE
        WHERE UPPER(username) = UPPER(:username)
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, username=username)
                row = cursor.fetchone()
    except Exception as exc:
        logger.warning(
            "Unable to resolve profile for user=%s. Using default=%s. error=%s",
            username,
            _default,
            exc,
        )
        return _default

    if not row:
        logger.info(
            "No profile found for user=%s. Using default=%s", username, _default
        )
        return _default

    profile_code, enabled = row[0], row[1]
    if int(enabled or 0) != 1:
        logger.info(
            "Profile disabled for user=%s. Using default=%s",
            username,
            _default,
        )
        return _default

    profile = str(profile_code or "").strip().upper()
    if profile not in ALLOWED_PROFILES:
        logger.warning(
            "Invalid profile_code=%s for user=%s. Using default=%s",
            profile_code,
            username,
            _default,
        )
        return _default

    return profile
