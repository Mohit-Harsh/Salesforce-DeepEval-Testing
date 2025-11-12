from InquirerPy import inquirer
from rich.console import Console
import os
from InquirerPy.base.control import Choice
import requests
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import typer
import json
import time
import html
import re
from InquirerPy.validator import PathValidator
import pandas as pd
import numpy as np
import csv
from deepeval.test_case import LLMTestCase
from deepEval import deepEvalTest

console = Console()

def parse_inputs_from_xml(xml_content: str) -> dict:
    # Parse XML and handle namespaces
    ns = {'ns': 'http://soap.sforce.com/2006/04/metadata'}
    root = ET.fromstring(xml_content)

    inputs_dict = {}

    # Iterate through <inputs> elements
    for inp in root.findall(".//ns:inputs", ns):
        api_name = inp.find("ns:apiName", ns).text if inp.find("ns:apiName", ns) is not None else None
        definition = inp.find("ns:definition", ns).text if inp.find("ns:definition", ns) is not None else None
        master_label = inp.find("ns:masterLabel", ns).text if inp.find("ns:masterLabel", ns) is not None else None
        reference_name = inp.find("ns:referenceName", ns).text if inp.find("ns:referenceName", ns) is not None else None
        required = inp.find("ns:required", ns).text.lower() == "true" if inp.find("ns:required", ns) is not None else False

        if api_name:
            inputs_dict[api_name] = {
                "definition": definition,
                "masterLabel": master_label,
                "referenceName": reference_name,
                "required": required
            }

    return inputs_dict

def getPromptList(creds):

    # Construct the queryAll endpoint URL
    url = f"{creds['instance_url']}/services/data/v62.0/einstein/prompt-templates"

    # Set up headers with authorization
    headers = {
        "Authorization": f"Bearer {creds['access_token']}",
        "Content-Type": "application/json"
    }

    # Send GET request with query as a parameter
    response = requests.get(url, headers=headers)

    json_response = response.json()

    json_response['promptRecords'] = list(filter(lambda x:x['isStandard'] is False,json_response['promptRecords']))

    result = list(map(lambda x:Choice(name=x['fields']['MasterLabel']['value'],value=x['fields']['DeveloperName']['value']),json_response['promptRecords']))
    
    return result

def getPromptTemplateInputs(apiName):

    dir = os.path.join(os.getcwd(),f'force-app/main/default/genAiPromptTemplates/{apiName}.genAiPromptTemplate-meta.xml')

    with Progress(SpinnerColumn(),TextColumn("[progress.description]{task.description}"),transient=True) as progress:

        progress.add_task(description="Retrieving Prompt Template Metadata...",total=None)
        result = subprocess.run(
            ["sf", "project", "retrieve", "start", "--metadata",f"GenAiPromptTemplate:{apiName}","--json"],
            capture_output=True,
            text=True,
            check=False,
            shell=True  # Do not raise error on non-zero exit code
        )
        
    if result.returncode != 0:

        console.print("Error querying prompt template metadata")

        raise typer.Exit(code=1)

    with open(dir, "r", encoding="utf-8") as f:
        xml_content = f.read()

    inputs = parse_inputs_from_xml(xml_content)

    return inputs


def parseJsonRequestBody(promptInputs, userInputs):
    """
    Build JSON request body for the Einstein Prompt Template Generations API.
    """
    value_map = {}

    # Iterate through prompt inputs and match with user-provided values
    for input_api_name, input_details in promptInputs.items():
        definition = input_details.get("definition", "")
        user_value = userInputs.get(input_api_name)

        # Determine input type
        if definition.startswith("SOBJECT://"):
            # Extract SObject API name (for reference only)
            sobject_name = definition.split("SOBJECT://")[-1]
            value_map[f"Input:{sobject_name}"] = {
                "value": {"id": user_value}
            }
        elif definition.startswith("primitive://"):
            value_map[f"Input:{input_api_name}"] = {
                "value": user_value
            }
        else:
            raise ValueError(f"Unknown definition type for {input_api_name}: {definition}")

    # Construct full JSON request body
    body = {
        "isPreview": False,
        "inputParams": {
            "valueMap": value_map
        },
        "additionalConfig": {
            "applicationName": "PromptTemplateGenerationsInvocable"
        }
    }

    return body


def promptPreviewRequest(instance_url, access_token, payload, api_name):
    """
    Send HTTP POST request to Einstein Prompt Template Generations endpoint.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    url = f"{instance_url}/services/data/v60.0/einstein/prompt-templates/{api_name}/generations"

    # Create a nice progress spinner while waiting for the API response
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task("[bold cyan]Generating response from Einstein Prompt Template...", start=False)
        progress.start_task(task)

        try:
            # Simulate "thinking" progress while waiting for API
            for _ in range(3):
                time.sleep(0.6)
                progress.advance(task, 20)

            # Send API request
            response = requests.post(url, headers=headers, data=json.dumps(payload))

            # Simulate a short delay to complete progress bar nicely
            for _ in range(2):
                time.sleep(0.5)
                progress.advance(task, 20)
        
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return None

    # Return parsed response or raise error if failed
    if response.status_code >= 200 and response.status_code < 300:
        return response.json()
    else:
        raise Exception(f"Request failed ({response.status_code}): {response.text}")
    

def promptPreview(creds,promptTemplate):

    inputs = getPromptTemplateInputs(promptTemplate)

    userInput = defaultdict()

    # Display a decorative info panel
    message = Text()
    message.append(f"\nYou have selected: ", style="bold white")
    message.append(f"{promptTemplate}\n\n", style="bold cyan")
    message.append("Enter all the required ", style="white")
    message.append("Prompt Template Input Fields ", style="bold yellow")
    message.append("as defined in your Salesforce template.\n\n", style="white")
    message.append("Please ensure that each field is filled accurately ", style="dim white")
    message.append("to generate the correct preview output.\n", style="dim white")

    console.print(
        Panel(
            message,
            title="[bold blue]Prompt Template Preview[/bold blue]",
            subtitle="Input Collection Stage 🧠",
            border_style="cyan",
            padding=(1, 2)
        )
    )

    for k in inputs.keys():

        instruction = f"({inputs[k]['definition']})"
        if inputs[k]['required']:
            instruction += ' (required)'
        userInput[k] = inquirer.text(message=f"{inputs[k]['masterLabel']}:",instruction=instruction).execute()


    jsonRequest = parseJsonRequestBody(promptInputs=inputs, userInputs=userInput)

    jsonResponse = promptPreviewRequest(instance_url=creds['instance_url'],access_token=creds['access_token'],payload=jsonRequest,api_name=promptTemplate)

    generation = jsonResponse['generations'][0]

    console.rule("[bold blue]Request Metadata")
    meta_table = Table(show_header=False)
    meta_table.add_row("Response ID", generation['responseId'])
    meta_table.add_row("Prompt Template Dev Name", jsonResponse['promptTemplateDevName'])
    meta_table.add_row("Request ID", jsonResponse['requestId'])
    meta_table.add_row("Finish Parameters", jsonResponse['parameters'])
    console.print(meta_table)

    console.rule("[bold cyan]Generated Response")
    console.print(
        Panel(
            generation['text'],
            title="[bold green]AI Response[/bold green]",
            border_style="green"
        )
    )

    console.rule("[bold yellow]Safety & Content Quality Scores")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Score")

    for k, v in generation['safetyScoreRepresentation'].items():
        table.add_row(k, f"{v:.6f}")

    toxicity_detected = generation['contentQualityRepresentation']['isToxicityDetected']
    table.add_row("Toxicity Detected", str(toxicity_detected))
    console.print(table)


def parseRetrievalContext(prompt):

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


def promptTestSingle(creds,promptTemplate):

    promptInputs = getPromptTemplateInputs(apiName=promptTemplate)
    
    userInputs = defaultdict()

    for k in promptInputs.keys():

        instruction = f"({promptInputs[k]['definition']})"
        if promptInputs[k]['required']:
            instruction += ' (required)'
        userInputs[k] = inquirer.text(message=f"{promptInputs[k]['masterLabel']}:",instruction=instruction).execute()

    expected_output = inquirer.text(message=f"Expected Output: ",instruction='(required)').execute()

    payload = parseJsonRequestBody(promptInputs=promptInputs,userInputs=userInputs)
            
    jsonResponse = promptTemplateExecute(instance_url=creds['instance_url'],access_token=creds['access_token'],payload=payload,api_name=promptTemplate)

    generation = jsonResponse['generations'][0]

    actual_response = generation['text']

    prompt = jsonResponse['prompt']

    retrieval_context = parseRetrievalContext(prompt)

    testCases = [LLMTestCase(
        input=str(userInputs),
        actual_output=actual_response,
        retrieval_context=retrieval_context,
        expected_output=expected_output
    )]

    with Progress(SpinnerColumn(),TextColumn("[progress.description]{task.description}"),transient=True) as progress:

        progress.add_task(description="Running Test Case...",total=None)
        parsedResults = deepEvalTest(testCases=testCases)

    metrics = parsedResults[0]['metrics']

    table = Table(title="Evaluation Metrics Summary")

    table.add_column("Metric Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Score", justify="center", style="green")
    table.add_column("Threshold", justify="center", style="yellow")
    table.add_column("Success", justify="center", style="bold")
    table.add_column("Model", justify="left", style="magenta")
    table.add_column("Reason", justify="left", style="white")

    for metric in metrics:
        
        table.add_row(
            metric["metric_name"],
            str(metric["score"]),
            str(metric["threshold"]),
            "✅" if metric["success"] else "❌",
            metric["evaluation_model"],
            metric["reason"]
        )

    console.print(table)


def promptTest(creds,promptTemplate):

    promptInputs = getPromptTemplateInputs(apiName=promptTemplate)

    home_path = "~/" if os.name == "posix" else "C:\\"
    src_path = inquirer.filepath(
        message="Enter file to upload:",
        default=home_path,
        validate=lambda path:True if (os.path.isfile(path) and path.lower().endswith(".csv")) else False,
        only_files=True,
    ).execute()

    # Read CSV and build structured dicts per row
    value_maps = []

    with open(src_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            value_map = {}
            
            for k,input_def in promptInputs.items():
                api_name = k
                definition = input_def["definition"]
                user_value = row.get(api_name, "").strip()

                # Skip empty optional fields
                if not user_value and not input_def.get("required", False):
                    continue

                if definition.startswith("SOBJECT://"):
                    sobject_name = definition.split("SOBJECT://")[-1]
                    value_map[f"Input:{sobject_name}"] = {
                        "value": {"id": user_value}
                    }
                elif definition.startswith("primitive://"):
                    value_map[f"Input:{api_name}"] = {
                        "value": user_value
                    }

            value_maps.append(value_map)

    console.print('CSV File Read: \n')
    console.print_json(value_maps)