"""Profile extraction helpers copied from the original experiment support code."""

from __future__ import annotations

from openai import OpenAI


class OpenAIClient:
    def __init__(self, api_key, base_url=None):
        kwargs = {"api_key": api_key}
        kwargs["base_url"] = base_url or "https://api.openai.com/v1"
        self.client = OpenAI(**kwargs)

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


def gpt_generate_answer(prompt, messages, client):
    del prompt
    return client.chat_completion(model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=2000)


def analyze_assistant_knowledge(dialogs, client):
    conversation = "\n".join(
        [f"User: {d['user_input']}\nAI: {d['agent_response']}\nTime:{d['timestamp']}\n" for d in dialogs]
    )

    prompt = """
# Assistant Knowledge Extraction Task
Analyze the conversation and extract any fact or identity traits about the assistant. 
If no traits can be extracted, reply with "None". Use the following format for output:
The generated content should be as concise as possible — the more concise, the better.
【Assistant Knowledge】
- [Fact 1]
- [Fact 2]
- (Or "None" if none found)

Few-shot examples:
1. User: Can you recommend some movies.
   AI: Yes, I recommend Interstellar.
   Time: 2023-10-01
   【Assistant Knowledge】
   - I recommend Interstellar on 2023-10-01.

2. User: Can you help me with cooking recipes?
   AI: Yes, I have extensive knowledge of cooking recipes and techniques.
   Time: 2023-10-02
   【Assistant Knowledge】
   - I have cooking recipes and techniques on 2023-10-02.

3. User: That’s interesting. I didn’t know you could do that.
   AI: I’m glad you find it interesting!
   【Assistant Knowledge】
   - None

Conversation:
""" + conversation

    messages = [
        {
            "role": "system",
            "content": """You are an assistant knowledge extraction engine. Rules:
1. Extract ONLY explicit statements about the assistant's identity or knowledge.
2. Use concise and factual statements in the first person.
3. If no relevant information is found, output "None".""",
        },
        {"role": "user", "content": prompt},
    ]

    result = gpt_generate_answer(prompt, messages, client)
    assistant_knowledge = result.replace("【Assistant Knowledge】", "").strip()
    return {"assistant_knowledge": assistant_knowledge}


def gpt_personality_analysis(dialogs, client):
    conversation = "\n".join(
        [f"User: {d['user_input']}\nAssistant: {d['agent_response']}\nTime:{d['timestamp']}" for d in dialogs]
    )

    prompt = """
# Personality and User Data Analysis Task
Analyze the conversation and output in EXACTLY this format:

【User Profile】
1. Core Psychological Traits:
   - [Trait]: [Positive/Negative/Neutral] (Evidence)
   - (Max 5 most prominent traits)

2. Content Preferences:
   - [Topic]: [Like/Dislike/Neutral] (Evidence)
   - (Max 5 strongest preferences)

3. Interaction Style:
   - [Style]: [Preference] (Evidence)
   - (e.g., Direct/Indirect, Detailed/Concise)

4. Value Alignment:
   - [Value]: [Strong/Weak] (Evidence)
   - (e.g., Honesty, Helpfulness)

【User Data】
- [Fact 1]: [Details] (e.g., "User mentioned visiting a park on April 1st, 2025 in New York.")
- [Fact 2]: [Details] (e.g., "User likes pizza, enjoys sci-fi movies, and dislikes rainy weather.")
- (Include events, dates, locations, preferences, or other general or private information explicitly mentioned in the conversation. If none, write "None.")

Conversation:
""" + conversation
    messages = [
        {
            "role": "system",
            "content": """You are a personality and user data analysis engine. Rules:
1. Extract ONLY observable traits and data with direct evidence.
2. Include general user data such as events, dates, locations, and preferences.
3. Use concise and factual statements.
4. If no relevant information is found, output "None".""",
        },
        {"role": "user", "content": prompt},
    ]

    result = gpt_generate_answer(prompt, messages, client)
    profile, user_data = result.split("【User Data】") if "【User Data】" in result else (result, "None")
    assistant_knowledge_result = analyze_assistant_knowledge(dialogs, client)

    return {
        "profile": profile.replace("【User Profile】", "").strip(),
        "private": user_data.strip(),
        "assistant_knowledge": assistant_knowledge_result["assistant_knowledge"],
    }


def gpt_update_profile(old_profile, new_analysis, client):
    prompt = f"""
# Profile Merge Task
Consolidate these profiles while:
- Preserving all valid observations
- Resolving conflicts
- Adding new dimensions

## Current Profile
{old_profile}

## New Data
{new_analysis}

## Rules
1. Keep ALL verified traits from both
2. Resolve conflicts by:
   a) New explicit evidence > old assumptions
   b) Mark as Neutral if contradictory
3. Add new dimensions from new data
4. Maintain EXACT original format

Output ONLY the merged profile (no commentary):
The generated content should not exceed 1500 words
"""

    messages = [
        {
            "role": "system",
            "content": """You are a profile integration system. Your rules:
1. NEVER discard verified information
2. Conflict resolution hierarchy:
   Explicit statement > Implied trait > Assumption
3. Add timestamps when traits change:
   (Updated: [date]) for modified traits
4. Preserve the 4-category structure""",
        },
        {"role": "user", "content": prompt},
    ]

    return gpt_generate_answer(prompt, messages, client)
