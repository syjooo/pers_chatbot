import openai
import os
import json

class MeritDescribeService:
    def __init__(self):
        base_path = os.path.join(os.path.dirname(__file__), "../resources")
        with open(os.path.join(base_path, "meritprompt.json"), "r", encoding="utf-8") as file:
            self.prompt_data = json.load(file)

    def analyze_team(self, team_scores: dict) -> str:
        system_prompt = self.prompt_data.get("system_prompt", "")
        team_scores_str = ", ".join([f"{k}: {v}" for k, v in team_scores.items()])
        full_prompt = f"{system_prompt}\n\nTeam Scores: {team_scores_str}"

        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": full_prompt}]
        )
        return response["choices"][0]["message"]["content"]
