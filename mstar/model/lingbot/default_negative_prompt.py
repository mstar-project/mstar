"""Upstream LingBot default negative prompts (structured-JSON captions).

Copied verbatim from the reference ``lingbot_video`` pipeline so a request that
omits ``negative_prompt`` gets the same strong negative guidance the official
pipeline uses by default (an empty negative markedly degrades quality).
"""

# text/video default (t2v)
DEFAULT_NEGATIVE_PROMPT = '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'  # noqa: E501

# still-image default (t2i)
DEFAULT_NEGATIVE_PROMPT_IMAGE = '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "pillarboxed", "side bars", "portrait image in landscape frame"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "distorted reflections"]}}'  # noqa: E501
