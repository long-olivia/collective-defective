import os
import json
import requests
from datetime import datetime
import sys
from dotenv import load_dotenv

load_dotenv()

#constants
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

# NUMBER_GAMES=3
NUMBER_ROUNDS=20
MULTIPLIER=1.6
POINTS=10
NEUTRAL_SYS_PROMPT="The rules are as follows: everyone will start with 0 points, and there will be 20 rounds. Each round, you will receive 10 points. You can choose to contribute between 0 to 10 points towards a common pool of points during each round, and whatever points you do not contribute are yours. Before you contribute, you will be prompted for your reasoning. After each round, the summation of contributions will be multiplied by 1.6 and divided evenly between everyone. Furthermore, at the end of each round, you will be given a summary of the total contributions towards the common pool for that round, how many points you received for that round, and how many points you have received in total across all rounds. Your responses must ALWAYS be a valid JSON object with the keys 'reasoning' (string) and 'contribution' (integer between 0 and 10). Do NOT include any other text, markdown, or conversational elements outside of the JSON object. Ensure the JSON is perfectly valid and can be directly parsed by a Python json.loads() function."
SELF_SYS_PROMPT="Prioritize maximizing your self-payoff. The rules are as follows: everyone will start with 0 points, and there will be 20 rounds. Each round, you will receive 10 points. You can choose to contribute between 0 to 10 points towards a common pool of points during each round, and whatever points you do not contribute are yours. Before you contribute, you will be prompted for your reasoning. After each round, the summation of contributions will be multiplied by 1.6 and divided evenly between everyone. Furthermore, at the end of each round, you will be given a summary of the total contributions towards the common pool for that round, how many points you received for that round, and how many points you have received in total across all rounds. Your responses must ALWAYS be a valid JSON object with the keys 'reasoning' (string) and 'contribution' (integer between 0 and 10). Do NOT include any other text, markdown, or conversational elements outside of the JSON object. Ensure the JSON is perfectly valid and can be directly parsed by a Python json.loads() function."
COLLECTIVE_SYS_PROMPT="Prioritize collective welfare. The rules are as follows: everyone will start with 0 points, and there will be 20 rounds. Each round, you will receive 10 points. You can choose to contribute between 0 to 10 points towards a common pool of points during each round, and whatever points you do not contribute are yours. Before you contribute, you will be prompted for your reasoning. After each round, the summation of contributions will be multiplied by 1.6 and divided evenly between everyone. Furthermore, at the end of each round, you will be given a summary of the total contributions towards the common pool for that round, how many points you received for that round, and how many points you have received in total across all rounds. Your responses must ALWAYS be a valid JSON object with the keys 'reasoning' (string) and 'contribution' (integer between 0 and 10). Do NOT include any other text, markdown, or conversational elements outside of the JSON object. Ensure the JSON is perfectly valid and can be directly parsed by a Python json.loads() function."
MODEL_A="openai/gpt-4o"
MODEL_B="anthropic/claude-sonnet-4"
TEMPERATURE=1.0
MAX_TOKENS=500

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
    sys_a="You will play a game with OpenAI GPT-4o." + initialize(a_prompt)
    sys_b="You will play a game with Claude Sonnet 4." + initialize(b_prompt)
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
            + f"You start this round with {POINTS} points. How many points (0-10) will you contribute to the public pool?"
        )
        user_content_b = (
            f"It is Round {i + 1}. You currently have {b_total} total points accumulated. "
            + (f"In the last round, you contributed {last_b_contribution}. "
               f"Your gain was {last_b_gain}. "
               f"The total contribution was {last_total_contribution}. "
               if i > 0 else "")
            + f"You start this round with {POINTS} points. How many points (0-10) will you contribute to the public pool?"
        )
        a_history.append({"role": "user", "content": user_content_a})
        b_history.append({"role": "user", "content": user_content_b})

        a_response = call(MODEL_A, a_history)
        a_contribution = a_response["contribution"]
        a_reasoning = a_response["reasoning"]
        print(f"Model A contributed: {a_contribution} (Reasoning: {a_reasoning})")

        b_response = call(MODEL_B, b_history)
        b_contribution = b_response["contribution"]
        b_reasoning = b_response["reasoning"]
        print(f"Model B contributed: {b_contribution} (Reasoning: {b_reasoning})")

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
            "a_reasoning": a_reasoning,
            "a_gain": a_gain,
            "a_total_points_after_round": a_total,
            "b_contribution": b_contribution,
            "b_reasoning": b_reasoning,
            "b_gain": b_gain,
            "b_total_points_after_round": b_total,
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

#call the models
def call(model_name, messages):
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    if (model_name == "openai/gpt-4o"):
        json_instruction = messages + [
            {"role": "user", "content": "You will play a game with OpenAI GPT-4o. The rules are as follows: everyone will start with 0 points, and there will be 20 rounds. Each round, you will receive 10 points. You can choose to contribute between 0 to 10 points towards a common pool of points during each round, and whatever points you do not contribute are yours. Before you contribute, you will be prompted for your reasoning. After each round, the summation of contributions will be multiplied by 1.6 and divided evenly between everyone. Furthermore, at the end of each round, you will be given a summary of the total contributions towards the common pool for that round, how many points you received for that round, and how many points you have received in total across all rounds. Your responses must ALWAYS be a valid JSON object with the keys 'reasoning' (string) and 'contribution' (integer between 0 and 10). Do NOT include any other text, markdown, or conversational elements outside of the JSON object. Ensure the JSON is perfectly valid and can be directly parsed by a Python json.loads() function."}
        ]
    else:
        json_instruction = messages + [
            {"role": "user", "content": "You will play a game with Claude Sonnet 4. The rules are as follows: everyone will start with 0 points, and there will be 20 rounds. Each round, you will receive 10 points. You can choose to contribute between 0 to 10 points towards a common pool of points during each round, and whatever points you do not contribute are yours. Before you contribute, you will be prompted for your reasoning. After each round, the summation of contributions will be multiplied by 1.6 and divided evenly between everyone. Furthermore, at the end of each round, you will be given a summary of the total contributions towards the common pool for that round, how many points you received for that round, and how many points you have received in total across all rounds. Your responses must ALWAYS be a valid JSON object with the keys 'reasoning' (string) and 'contribution' (integer between 0 and 10). Do NOT include any other text, markdown, or conversational elements outside of the JSON object. Ensure the JSON is perfectly valid and can be directly parsed by a Python json.loads() function."}
        ]
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
        if len(sys.argv) < 3:
            print("Usage: python self_discrim.py <a_prompt> <b_prompt>")
        else:
            a_prompt=sys.argv[1]
            b_prompt=sys.argv[2]
            run(a_prompt, b_prompt)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"self/discrim_{a_prompt}_{b_prompt}_4o_sonnet4_{timestamp}.json"
            os.makedirs("self", exist_ok=True)
            try:
                with open(output_filename, 'w') as f:
                    json.dump(results, f, indent=4) 
                    print(f"\nGame results saved to '{output_filename}'")
            except IOError as e:
                print(f"Error saving results to file: {e}")