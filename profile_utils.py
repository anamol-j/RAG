def normalize_preferences(raw: dict):
    """
    Normalize randomly selected survey answers into a stable profile.
    Missing questions are safely defaulted.
    """

    return {
        # Identity
        "gender": raw.get("gender", None),
        "name": raw.get("name", None),
        "age_range": raw.get("age_range", None),

        # Body attributes
        "body_shape": raw.get("body_shape", None),
        "body_size": raw.get("body_size", None),
        "skin_tone": raw.get("skin_tone", None),

        # Style intent
        "style_goal": raw.get("style_goal", []),
        "difficult_occasion": raw.get("difficult_occasion", []),
        "pick_style": raw.get("pick_style", []),
        "style_preference": raw.get("style_preference", []),

        # Fit & aesthetics
        "preferred_fit": raw.get("preferred_fit", None),
        "outfit_boldness": raw.get("outfit_boldness", None),
        "color_comfort": raw.get("color_comfort", None),
    }
