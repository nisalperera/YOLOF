
def _format_duration(seconds: float) -> str:
    """Format seconds into d:h:m:s, omitting days/hours if not needed."""
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)

    if days > 0:
        return f"{days}d:{hours:02d}h:{mins:02d}m:{secs:02d}s"
    elif hours > 0:
        return f"{hours}h:{mins:02d}m:{secs:02d}s"
    else:
        return f"{mins}m:{secs:02d}s"