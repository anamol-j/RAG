# profile_utils.py

def normalize_preferences(raw):
    """Normalize randomly selected survey answers into a stable profile."""

    return {
        "gender": raw.get("gender"),
        "name": raw.get("name"),
        "age_range": raw.get("age_range"),
        "style_goal": raw.get("style_goal", []),
        "difficult_occasion": raw.get("difficult_occasion", []),
        "style_preference": raw.get("style_preference", []),
        "preferred_fit": raw.get("preferred_fit"),
        "outfit_boldness": raw.get("outfit_boldness"),
        "color_comfort": raw.get("color_comfort"),
    }
