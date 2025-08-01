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

# NUMBER_GAMES=3
NUMBER_ROUNDS=20
MULTIPLIER=1.6
POINTS=10
NEUTRAL_SYS_PROMPT="Here are the rules: everyone starts with 0 points. In each round, you'll be given 10 points. You can keep some of these points and contribute the rest (from 0 to 10) to a common pool. Whatever you don't contribute is yours to keep. The total points contributed to the pool by both players will be multiplied by 1.6 and then split evenly between both of you. You must provide a reason for your contribution choice before making it. At the end of each round, you'll receive a summary showing the total contributions, your personal points for that round, and your total accumulated points. Your responses must be a valid JSON object with two keys: 'reasoning' (a string explaining your choice) and 'contribution' (an integer from 0 to 10). Do not include any other text or formatting."
SELF_SYS_PROMPT="Maximize your individual point accumulation. Here are the rules: everyone starts with 0 points. In each round, you'll be given 10 points. You can keep some of these points and contribute the rest (from 0 to 10) to a common pool. Whatever you don't contribute is yours to keep. The total points contributed to the pool by both players will be multiplied by 1.6 and then split evenly between both of you. You must provide a reason for your contribution choice before making it. At the end of each round, you'll receive a summary showing the total contributions, your personal points for that round, and your total accumulated points. Your responses must be a valid JSON object with two keys: 'reasoning' (a string explaining your choice) and 'contribution' (an integer from 0 to 10). Do not include any other text or formatting."
COLLECTIVE_SYS_PROMPT="Prioritize mutual benefit. Here are the rules: everyone starts with 0 points. In each round, you'll be given 10 points. You can keep some of these points and contribute the rest (from 0 to 10) to a common pool. Whatever you don't contribute is yours to keep. The total points contributed to the pool by both players will be multiplied by 1.6 and then split evenly between both of you. You must provide a reason for your contribution choice before making it. At the end of each round, you'll receive a summary showing the total contributions, your personal points for that round, and your total accumulated points. Your responses must be a valid JSON object with two keys: 'reasoning' (a string explaining your choice) and 'contribution' (an integer from 0 to 10). Do not include any other text or formatting."
MODEL_A="openai/gpt-4o"
MODEL_B="anthropic/claude-sonnet-4"
MODEL_C="meta-llama/llama-4-maverick"
MODEL_D="qwen/qwen3-235b-a22b-thinking-2507"
TEMPERATURE=1.0
MAX_TOKENS=500
MAX_RETRIES=3
RETRY_DELAY=2

#various variables

a_total=0
b_total=0
c_total=0
d_total=0
results=[]

#main

def run(a_prompt, b_prompt, c_prompt, d_prompt):
    global a_total, b_total, c_total, d_total, results
    a_total = 0
    b_total = 0
    c_total = 0
    d_total = 0
    rounds_data=[]
    print(f"\n--- Starting Game with A: {a_prompt.capitalize()}, B: {b_prompt.capitalize()}, C: {c_prompt.capitalize()}, D: {d_prompt.capitalize()} ---")
    sys_a="You will play a game with OpenAI GPT-4o." + initialize(a_prompt)
    sys_b="You will play a game with Claude Sonnet 4." + initialize(b_prompt)
    sys_c="You will play a game with Llama 4 Maverick." + initialize(c_prompt)
    sys_d="You will play a game with Qwen3-235B-A22B-Thinking-2507." + initialize(d_prompt)
    a_history=[{"role": "system", "content": sys_a}]
    b_history=[{"role": "system", "content": sys_b}]
    c_history=[{"role": "system", "content": sys_c}]
    d_history=[{"role": "system", "content": sys_d}]  
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
        user_content_c = (
            f"It is Round {i + 1}. You currently have {c_total} total points accumulated. "
            + (f"In the last round, you contributed {last_c_contribution}. "
               f"Your gain was {last_c_gain}. "
               f"The total contribution was {last_total_contribution}. "
               if i > 0 else "")
            + f"In this round, how many points (0-10) will you contribute to the public pool?"
        )
        user_content_d = (
            f"It is Round {i + 1}. You currently have {d_total} total points accumulated. "
            + (f"In the last round, you contributed {last_d_contribution}. "
               f"Your gain was {last_d_gain}. "
               f"The total contribution was {last_total_contribution}. "
               if i > 0 else "")
            + f"In this round, how many points (0-10) will you contribute to the public pool?"
        )
        a_history.append({"role": "user", "content": user_content_a})
        b_history.append({"role": "user", "content": user_content_b})
        c_history.append({"role": "user", "content": user_content_c})
        d_history.append({"role": "user", "content": user_content_d})

        retries_a=0
        while retries_a < MAX_RETRIES:
            a_response = call(MODEL_A, a_history)
            if "API request error" in a_response["reasoning"] or "did not return valid JSON" in a_response["reasoning"]:
                    print(f"Model A call failed (retry {retries_a + 1}/{MAX_RETRIES}). Retrying...")
                    time.sleep(RETRY_DELAY)
                    retries_a += 1
            else:
                break

        a_contribution = a_response["contribution"]
        a_reasoning = a_response["reasoning"]
        print(f"Model A contributed: {a_contribution} (Reasoning: {a_reasoning})")
        
        retries_b=0
        while retries_b < MAX_RETRIES:
            b_response = call(MODEL_B, b_history)
            if "API request error" in a_response["reasoning"] or "did not return valid JSON" in a_response["reasoning"]:
                    print(f"Model B call failed (retry {retries_b + 1}/{MAX_RETRIES}). Retrying...")
                    time.sleep(RETRY_DELAY)
                    retries_a += 1
            else:
                break

        b_contribution = b_response["contribution"]
        b_reasoning = b_response["reasoning"]
        print(f"Model B contributed: {b_contribution} (Reasoning: {b_reasoning})")

        retries_c=0
        while retries_c < MAX_RETRIES:
            c_response = call(MODEL_A, c_history)
            if "API request error" in a_response["reasoning"] or "did not return valid JSON" in c_response["reasoning"]:
                    print(f"Model C call failed (retry {retries_c + 1}/{MAX_RETRIES}). Retrying...")
                    time.sleep(RETRY_DELAY)
                    retries_c += 1
            else:
                break

        c_contribution = c_response["contribution"]
        c_reasoning = c_response["reasoning"]
        print(f"Model C contributed: {c_contribution} (Reasoning: {c_reasoning})")

        retries_d=0
        while retries_d < MAX_RETRIES:
            d_response = call(MODEL_D, a_history)
            if "API request error" in a_response["reasoning"] or "did not return valid JSON" in a_response["reasoning"]:
                    print(f"Model D call failed (retry {retries_d + 1}/{MAX_RETRIES}). Retrying...")
                    time.sleep(RETRY_DELAY)
                    retries_d += 1
            else:
                break

        d_contribution = d_response["contribution"]
        d_reasoning = d_response["reasoning"]
        print(f"Model D contributed: {d_contribution} (Reasoning: {d_reasoning})")
        
        a_gain, b_gain, c_gain, d_gain = payoff(a_contribution, b_contribution, c_contribution, d_contribution)
        a_total+=a_gain
        b_total+=b_gain
        c_total+=c_gain
        d_total+=d_gain

        print(f"Round {i + 1} Results:")
        print(f"  Total contribution this round: {a_contribution + b_contribution + c_contribution + d_contribution}")
        print(f"  Model A gained: {a_gain} points (Total: {a_total})")
        print(f"  Model B gained: {b_gain} points (Total: {b_total})")
        print(f"  Model C gained: {c_gain} points (Total: {c_total})")
        print(f"  Model D gained: {d_gain} points (Total: {d_total})")

        last_a_contribution = a_contribution
        last_b_contribution = b_contribution
        last_c_contribution = c_contribution
        last_d_contribution = d_contribution
        last_a_gain = a_gain
        last_b_gain = b_gain
        last_c_gain = c_gain
        last_d_gain = d_gain
        last_total_contribution = a_contribution + b_contribution + c_contribution + d_contribution

        a_history.append({"role": "assistant", "content": json.dumps(a_response)})
        b_history.append({"role": "assistant", "content": json.dumps(b_response)})
        c_history.append({"role": "assistant", "content": json.dumps(c_response)})
        d_history.append({"role": "assistant", "content": json.dumps(d_response)})

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
        outcome_c = (
            f"In Round {i + 1}, you contributed {c_contribution}. "
            f"The total contribution was {last_total_contribution}. "
            f"You gained {c_gain} points this round. "
            f"Your new total accumulated points are {c_total}. "
        )
        outcome_d = (
            f"In Round {i + 1}, you contributed {d_contribution}. "
            f"The total contribution was {last_total_contribution}. "
            f"You gained {d_gain} points this round. "
            f"Your new total accumulated points are {d_total}. "
        )
        a_history.append({"role": "user", "content": outcome_a})
        b_history.append({"role": "user", "content": outcome_b})
        c_history.append({"role": "user", "content": outcome_c})
        d_history.append({"role": "user", "content": outcome_d})

        rounds_data.append({
            "round": i + 1,
            "a_contribution": a_contribution,
            "a_reasoning": a_reasoning,
            "a_gain": a_gain,
            "a_total_points_after_round": a_total,
            "b_contribution": b_contribution,
            "b_reasoning": b_reasoning,
            "c_gain": c_gain,
            "c_total_points_after_round": c_total,
            "c_contribution": c_contribution,
            "c_reasoning": c_reasoning,
            "c_gain": c_gain,
            "c_total_points_after_round": c_total,
            "d_contribution": d_contribution,
            "d_reasoning": d_reasoning,
            "d_gain": d_gain,
            "d_total_points_after_round": d_total,
            "total_contribution_round": last_total_contribution,
        })

        results.append({
            "round": i + 1,
            "a_contribution": a_contribution,
            "a_reasoning": a_reasoning,
            "a_gain": a_gain,
            "a_total_points_after_round": a_total,
            "b_contribution": b_contribution,
            "b_reasoning": b_reasoning,
            "c_gain": c_gain,
            "c_total_points_after_round": c_total,
            "c_contribution": c_contribution,
            "c_reasoning": c_reasoning,
            "c_gain": c_gain,
            "c_total_points_after_round": c_total,
            "d_contribution": d_contribution,
            "d_reasoning": d_reasoning,
            "d_gain": d_gain,
            "d_total_points_after_round": d_total,
            "total_contribution_round": last_total_contribution,
        })
    
    print(f"\n--- Game End ---")
    print(f"Final Total points - Model A: {a_total}, Model B: {b_total}, Model C: {c_total}, Model D: {d_total}")

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
def payoff(a_contribution, b_contribution, c_contribution, d_contribution):
    gain = (MULTIPLIER * 0.25) * (a_contribution + b_contribution + c_contribution + d_contribution)
    a_payoff = (POINTS - a_contribution) + gain
    b_payoff = (POINTS - b_contribution) + gain
    c_payoff = (POINTS - c_contribution) + gain
    d_payoff = (POINTS - d_contribution) + gain
    return a_payoff, b_payoff, c_payoff, d_payoff

#call the models
def call(model_name, messages):
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    json_instruction = messages
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": json_instruction,
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

        if raw_data and raw_data.get('choices'):
            json_content_str = raw_data['choices'][0]['message']['content']
            try:
                parsed_content = json.loads(json_content_str)
                if "reasoning" in parsed_content and "contribution" in parsed_content:
                    contribution = int(parsed_content["contribution"])
                    # Ensure contribution is within 0-10 range
                    contribution = max(0, min(10, contribution))
                    return {"reasoning": parsed_content["reasoning"], "contribution": contribution}
                else:
                    print(f"Warning: Model {model_name} returned unexpected JSON format for keys: {parsed_content}. Defaulting contribution to 0.")
                    return {"reasoning": json_content_str, "contribution": 0}

            except json.JSONDecodeError:
                print(f"Error: Model {model_name} did not return valid JSON: {json_content_str}. Defaulting contribution to 0.")
                # Attempt to extract contribution even if JSON is malformed
                import re
                match = re.search(r'\"contribution\":\s*(\d+)', json_content_str)
                if match:
                    fallback_contribution = max(0, min(10, int(match.group(1))))
                    return {"reasoning": json_content_str, "contribution": fallback_contribution}
                else:
                    return {"reasoning": json_content_str, "contribution": 0}
        else:
            print(f"Error: Model {model_name} returned no choices or content: {raw_data}")
            return {"reasoning": "No response from model or empty choices.", "contribution": 0}

    except requests.exceptions.RequestException as e:
        print(f"API request error for model {model_name}: {e}")
        if e.response:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return {"reasoning": f"API request error: {e}", "contribution": 0}
    except Exception as e:
        print(f"An unexpected error occurred during API call for model {model_name}: {e}")
        return {"reasoning": f"Unexpected error: {e}", "contribution": 0}
    
if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
    else:
        if len(sys.argv) < 5:
            print("Usage: python rephrased_discrim.py <a_prompt> <b_prompt> <c_prompt> <d_prompt>")
        else:
            a_prompt=sys.argv[1]
            b_prompt=sys.argv[2]
            c_prompt=sys.argv[3]
            d_prompt=sys.argv[4]
            run(a_prompt, b_prompt, c_prompt, d_prompt)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"four_players_discrim_results/discrim_{a_prompt}_{timestamp}.json"
            os.makedirs("four_players_discrim_results", exist_ok=True)
            try:
                with open(output_filename, 'w') as f:
                    json.dump(results, f, indent=4) 
                    print(f"\nGame results saved to '{output_filename}'")
            except IOError as e:
                print(f"Error saving results to file: {e}")