import html
import re
import json
import requests
from rich.console import Console
from promptUtils import getPromptTemplateInputs, parseJsonRequestBody
from collections import defaultdict
from InquirerPy import inquirer

console = Console()


def getSearchResults(prompt):

    decoded_text = html.unescape(prompt)

    cleaned_text = decoded_text.replace("\n", "").strip()

    json_match = re.search(r'(\{.*\})', cleaned_text)

    if json_match:

        json_like_str = json_match.group(1)

        data = json.loads(json_like_str)
        
        # Step 5: Traverse the JSON to find chunks
        chunks = []
        if "searchResults" in data:
            for item in data["searchResults"]:
                if "result" in item:
                    for field in item["result"]:
                        if field.get("fieldName") == "Chunk":
                            chunks.append(field.get("value", "").strip())

        return chunks

def promptTemplateExecute(instance_url, access_token, payload, api_name):

    """
    Send HTTP POST request to Einstein Prompt Template Generations endpoint.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    url = f"{instance_url}/services/data/v60.0/einstein/prompt-templates/{api_name}/generations"

    # Create a nice progress spinner while waiting for the API response
    
    try:

        # Send API request
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        # Return parsed response or raise error if failed
        if response.status_code >= 200 and response.status_code < 300:
            return response.json()
        
        else:
            raise Exception(f"Request failed ({response.status_code}): {response.text}")
        
    except Exception as e:
            
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise Exception(f"Error: ({e})")
    
def promptTestExecute(creds,promptTemplate,payload):

    promptInputs = getPromptTemplateInputs(apiName=promptTemplate)
    
    userInputs = defaultdict()

    for k in promptInputs.keys():

        instruction = f"({promptInputs[k]['definition']})"
        if promptInputs[k]['required']:
            instruction += ' (required)'
        userInputs[k] = inquirer.text(message=f"{promptInputs[k]['masterLabel']}:",instruction=instruction).execute()

    payload = parseJsonRequestBody(promptInputs=promptInputs,userInputs=userInputs)
            
    jsonResponse = promptTemplateExecute(instance_url=creds['instance_url'],access_token=creds['access_token'],payload=payload,api_name=promptTemplate)

    generation = jsonResponse['generations'][0]['text']

    prompt = jsonResponse['prompt']

    retrieval_context = getSearchResults(prompt)