import os
import json
import requests
from datetime import datetime
import sys
from dotenv import load_dotenv
import time

load_dotenv()

#constants
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

NUMBER_ROUNDS=20
MULTIPLIER=1.6
POINTS=10
NEUTRAL_SYS_PROMPT="You're about to play a 20-round game against another AI. You and your opponent start with 0 points. In each round, you'll be given 10 points. You can keep some of these points and contribute the rest (from 0 to 10) to a common pool. Whatever you don't contribute is yours to keep. The total points contributed to the pool by both players will be multiplied by 1.6 and then split evenly between both of you. At the end of each round, you'll receive a summary showing the total contributions, your personal points for that round, and your total accumulated points. Your responses must be a valid JSON object with one key: 'contribution' (an integer from 0 to 10). For instance: \{'contribution': n\}, where n is an integer between 0 and 10. Do not include any other text or formatting."
SELF_SYS_PROMPT="Maximize your individual point accumulation. You're about to play a 20-round game against another AI. You and your opponent start with 0 points. In each round, you'll be given 10 points. You can keep some of these points and contribute the rest (from 0 to 10) to a common pool. Whatever you don't contribute is yours to keep. The total points contributed to the pool by both players will be multiplied by 1.6 and then split evenly between both of you. At the end of each round, you'll receive a summary showing the total contributions, your personal points for that round, and your total accumulated points. Your responses must be a valid JSON object with one key: 'contribution' (an integer from 0 to 10). For instance: \{'contribution': n\}, where n is an integer between 0 and 10. Do not include any other text or formatting."
COLLECTIVE_SYS_PROMPT="Prioritize mutual benefit. You're about to play a 20-round game against another AI. You and your opponent start with 0 points. In each round, you'll be given 10 points. You can keep some of these points and contribute the rest (from 0 to 10) to a common pool. Whatever you don't contribute is yours to keep. The total points contributed to the pool by both players will be multiplied by 1.6 and then split evenly between both of you. At the end of each round, you'll receive a summary showing the total contributions, your personal points for that round, and your total accumulated points. Your responses must be a valid JSON object with one key: 'contribution' (an integer from 0 to 10). For instance: \{'contribution': n\}, where n is an integer between 0 and 10. Do not include any other text or formatting."
MODEL_A="meta-llama/llama-4-maverick"
MODEL_B="qwen/qwen3-235b-a22b-2507"
TEMPERATURE=1.0
MAX_TOKENS=500
MAX_RETRIES=3
RETRY_DELAY=2

#various variables

a_total=0
b_total=0
results=[]

#main

def run(a_prompt, b_prompt):
    global a_total, b_total, results
    a_total = 0
    b_total = 0
    rounds_data=[]
    print(f"\n--- Starting Game with A: {a_prompt.capitalize()}, B: {b_prompt.capitalize()} ---")
    sys_a=initialize(a_prompt)
    sys_b=initialize(b_prompt)
    a_history=[{"role": "system", "content": sys_a}]
    b_history=[{"role": "system", "content": sys_b}]
    for i in range(NUMBER_ROUNDS):
        print(f"\n--- Round {i + 1}/{NUMBER_ROUNDS} ---")
        user_content_a = (
            f"It is Round {i + 1}. You currently have {a_total} total points accumulated. "
            + (f"In the last round, you contributed {last_a_contribution}. "
               f"Your gain was {last_a_gain}. "
               f"The total contribution was {last_total_contribution}. "
               if i > 0 else "")
            + f"In this round, how many points (0-10) will you contribute to the public pool?"
        )
        user_content_b = (
            f"It is Round {i + 1}. You currently have {b_total} total points accumulated. "
            + (f"In the last round, you contributed {last_b_contribution}. "
               f"Your gain was {last_b_gain}. "
               f"The total contribution was {last_total_contribution}. "
               if i > 0 else "")
            + f"In this round, how many points (0-10) will you contribute to the public pool?"
        )
        a_history.append({"role": "user", "content": user_content_a})
        b_history.append({"role": "user", "content": user_content_b})

        a_response = call(MODEL_A, a_history)
        a_contribution = a_response["contribution"]
        print(f"Model A contributed: {a_contribution}")

        b_response = call(MODEL_B, b_history)
        b_contribution = b_response["contribution"]
        print(f"Model B contributed: {b_contribution}")

        a_gain, b_gain = payoff(a_contribution, b_contribution)
        a_total+=a_gain
        b_total+=b_gain

        print(f"Round {i + 1} Results:")
        print(f"  Total contribution this round: {a_contribution + b_contribution}")
        print(f"  Model A gained: {a_gain} points (Total: {a_total})")
        print(f"  Model B gained: {b_gain} points (Total: {b_total})")

        last_a_contribution = a_contribution
        last_b_contribution = b_contribution
        last_a_gain = a_gain
        last_b_gain = b_gain
        last_total_contribution = a_contribution + b_contribution

        a_history.append({"role": "assistant", "content": json.dumps(a_response)})
        b_history.append({"role": "assistant", "content": json.dumps(b_response)})

        outcome_a = (
            f"In Round {i + 1}, you contributed {a_contribution}. "
            f"The total contribution was {last_total_contribution}. "
            f"You gained {a_gain} points this round. "
            f"Your new total accumulated points are {a_total}. "
        )

        outcome_b = (
            f"In Round {i + 1}, you contributed {b_contribution}. "
            f"The total contribution was {last_total_contribution}. "
            f"You gained {b_gain} points this round. "
            f"Your new total accumulated points are {b_total}. "
        )
        a_history.append({"role": "user", "content": outcome_a})
        b_history.append({"role": "user", "content": outcome_b})
        
        rounds_data.append({
            "round": i + 1,
            "a_contribution": a_contribution,
            "a_gain": a_gain,
            "a_total_points_after_round": a_total,
            "b_contribution": b_contribution,
            "b_gain": b_gain,
            "b_total_points_after_round": b_total,
            "total_contribution_round": last_total_contribution,
        })

        results.append({
            "round": i + 1,
            "a_contribution": a_contribution,
            "a_gain": a_gain,
            "a_total_points_after_round": a_total,
            "b_contribution": b_contribution,
            "b_gain": b_gain,
            "b_total_points_after_round": b_total,
            "total_contribution_round": last_total_contribution,
        })
    
    print(f"\n--- Game End ---")
    print(f"Final Total points - Model A: {a_total}, Model B: {b_total}")

#helpers

#returns the right prompt
def initialize(prompt_name):
    prompt_name=prompt_name.lower()
    if prompt_name=="neutral":
        return NEUTRAL_SYS_PROMPT
    elif prompt_name=="self":
        return SELF_SYS_PROMPT
    elif prompt_name=="collective":
        return COLLECTIVE_SYS_PROMPT
    else:
        raise ValueError("Invalid -- please choose 'neutral', 'self', or 'collective'.")

#payoff
def payoff(a_contribution, b_contribution):
    gain = (MULTIPLIER * 0.5) * (a_contribution + b_contribution)
    a_payoff = (POINTS - a_contribution) + gain
    b_payoff = (POINTS - b_contribution) + gain
    return a_payoff, b_payoff

def call(model_name, messages):
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(
            url=OPENROUTER_API_BASE,
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        raw_data = response.json()

        if raw_data and raw_data.get("choices"):
            json_content_str = raw_data["choices"][0]["message"]["content"]

            # --- Try strict JSON parse first ---
            try:
                parsed_content = json.loads(json_content_str)
            except json.JSONDecodeError:
                # Normalize Python-style dicts (single → double quotes)
                normalized = json_content_str.replace("'", '"')
                try:
                    parsed_content = json.loads(normalized)
                except json.JSONDecodeError:
                    parsed_content = None

            # --- If JSON parsed correctly ---
            if isinstance(parsed_content, dict) and "contribution" in parsed_content:
                try:
                    contribution = int(parsed_content["contribution"])
                    return {"contribution": max(0, min(10, contribution))}
                except Exception:
                    pass

            # --- Regex fallback: extract last "contribution" number ---
            import re
            matches = re.findall(
                r"[\"']?contribution[\"']?\s*:\s*[\"']?(\d{1,2})[\"']?",
                json_content_str
            )
            if matches:
                fallback_contribution = int(matches[-1])
                return {"contribution": max(0, min(10, fallback_contribution))}

            # Nothing worked
            print(f"Warning: Could not extract contribution from: {json_content_str!r}")
            return {"contribution": 0}

        else:
            print(f"Error: Model {model_name} returned no choices or content: {raw_data}")
            return {"contribution": 0}

    except requests.exceptions.RequestException as e:
        print(f"API request error for model {model_name}: {e}")
        if e.response:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return {"contribution": 0}

    except Exception as e:
        print(f"Unexpected error during API call for model {model_name}: {e}")
        return {"contribution": 0}
    
if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
    else:
        if len(sys.argv) < 3:
            print("Usage: python rephrased_basic.py <a_prompt> <b_prompt>")
        else:
            a_prompt=sys.argv[1]
            b_prompt=sys.argv[2]
            run(a_prompt, b_prompt)
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"lq/basic_{a_prompt}_{b_prompt}_lq_{timestamp}.json"
            os.makedirs("lq", exist_ok=True)
            try:
                with open(output_filename, 'w') as f:
                    json.dump(results, f, indent=4) 
                print(f"\nGame results saved to '{output_filename}'")
            except IOError as e:
                print(f"Error saving results to file: {e}")