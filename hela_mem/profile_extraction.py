"""Profile and knowledge extraction helpers used during encoding."""

from __future__ import annotations

from .utils import OpenAIChatClient, gpt_generate_answer, resolve_model_name


def analyze_assistant_knowledge(dialogs: list[dict[str, str]], client: OpenAIChatClient) -> dict[str, str]:
    conversation = "\n".join(
        f"User: {dialog['user_input']}\nAssistant: {dialog['agent_response']}\nTime: {dialog['timestamp']}"
        for dialog in dialogs
    )

    prompt = """
# Assistant Knowledge Extraction Task
Analyze the conversation and extract any explicit fact or identity trait about the assistant.
If nothing useful is present, return "None".

Output format:
【Assistant Knowledge】
- fact 1
- fact 2
- None

Conversation:
""" + conversation

    messages = [
        {
            "role": "system",
            "content": (
                "You extract explicit assistant facts only. Use concise factual statements "
                "in first person when possible."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    result = gpt_generate_answer(
        prompt,
        messages,
        client=client,
        model=resolve_model_name(),
        temperature=0.0,
        max_tokens=512,
    )
    return {"assistant_knowledge": result.replace("【Assistant Knowledge】", "").strip()}


def gpt_personality_analysis(dialogs: list[dict[str, str]], client: OpenAIChatClient) -> dict[str, str]:
    conversation = "\n".join(
        f"User: {dialog['user_input']}\nAssistant: {dialog['agent_response']}\nTime: {dialog['timestamp']}"
        for dialog in dialogs
    )

    prompt = """
# Personality and User Data Analysis Task
Analyze the conversation and output exactly in this format:

【User Profile】
1. Core Psychological Traits:
   - [Trait]: [Positive/Negative/Neutral] (Evidence)
2. Content Preferences:
   - [Topic]: [Like/Dislike/Neutral] (Evidence)
3. Interaction Style:
   - [Style]: [Preference] (Evidence)
4. Value Alignment:
   - [Value]: [Strong/Weak] (Evidence)

【User Data】
- [Fact 1]
- [Fact 2]
- None

Conversation:
""" + conversation

    messages = [
        {
            "role": "system",
            "content": (
                "You extract observable user traits and user facts with direct evidence only. "
                "Stay concise and factual."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    result = gpt_generate_answer(
        prompt,
        messages,
        client=client,
        model=resolve_model_name(),
        temperature=0.0,
        max_tokens=1800,
    )

    if "【User Data】" in result:
        profile, user_data = result.split("【User Data】", 1)
    else:
        profile, user_data = result, "None"

    assistant_knowledge = analyze_assistant_knowledge(dialogs, client)["assistant_knowledge"]
    return {
        "profile": profile.replace("【User Profile】", "").strip(),
        "private": user_data.strip(),
        "assistant_knowledge": assistant_knowledge,
    }


def gpt_update_profile(old_profile: str, new_analysis: str, client: OpenAIChatClient) -> str:
    prompt = f"""
# Profile Merge Task
Merge the current profile with the new observations.

## Current Profile
{old_profile}

## New Observations
{new_analysis}

Rules:
1. Keep verified information from both.
2. Resolve conflicts in favor of newer explicit evidence.
3. Preserve the four-section structure.
4. Output only the merged profile.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You merge user profiles without dropping verified facts. "
                "Prefer new explicit evidence over older assumptions."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    return gpt_generate_answer(
        prompt,
        messages,
        client=client,
        model=resolve_model_name(),
        temperature=0.0,
        max_tokens=1800,
    )
