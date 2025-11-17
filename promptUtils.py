from InquirerPy import inquirer
from rich.console import Console
import os
from InquirerPy.base.control import Choice
import requests
import subprocess
from rich import box
from collections import defaultdict
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.pretty import Pretty
from rich.align import Align
import typer
import json
import time
import html
import re
import concurrent.futures
import pandas as pd
import numpy as np
from deepeval.test_case import LLMTestCase
from deepEval import deepEvalTest
import xmltodict

console = Console()

def parseXmlMetadata(promptTemplate: str) -> dict:

    # Parse XML and handle namespaces

    dir = os.path.join(os.getcwd(),f'force-app/main/default/genAiPromptTemplates/{promptTemplate}.genAiPromptTemplate-meta.xml')

    with open(dir, "r", encoding="utf-8") as f:
        xml_content = f.read()

    metadata = xmltodict.parse(xml_content)

    if type(metadata['GenAiPromptTemplate']['templateVersions']) == dict:

        metadata['GenAiPromptTemplate']['templateVersions'] = [metadata['GenAiPromptTemplate']['templateVersions']]

    if type(metadata['GenAiPromptTemplate']['templateVersions'][0]['inputs']) == dict:

        metadata['GenAiPromptTemplate']['templateVersions'][0]['inputs'] = [metadata['GenAiPromptTemplate']['templateVersions'][0]['inputs']]
    
    return metadata

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
    
    templateVersions = parseXmlMetadata(promptTemplate=apiName)['GenAiPromptTemplate']['templateVersions']

    return templateVersions[0]['inputs']


def parseJsonRequestBody(promptInputs: list[dict], userInputs: dict):
    """
    Build JSON request body for the Einstein Prompt Template Generations API.
    """
    value_map = {}

    # Iterate through prompt inputs and match with user-provided values
    for input_details in promptInputs:
        definition = input_details.get("definition", "")
        user_value = userInputs.get(input_details.get('apiName'))

        # Determine input type
        if definition.startswith("SOBJECT://"):
            # Extract SObject API name (for reference only)
            sobject_name = definition.split("SOBJECT://")[-1]
            value_map[f"Input:{sobject_name}"] = {
                "value": {"id": user_value}
            }
        elif definition.startswith("primitive://"):
            value_map[f"Input:{input_details.get('apiName')}"] = {
                "value": user_value
            }
        else:
            raise ValueError(f"Unknown definition type for {input_details.get('apiName')}: {definition}")

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


def promptTemplateExecute(instance_url:str, access_token:str, payload:dict, api_name:str):
    """
    Send HTTP POST request to Einstein Prompt Template Generations endpoint.
    """

    headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

    url = f"{instance_url}/services/data/v60.0/einstein/prompt-templates/{api_name}/generations"

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
    
def simulatePromptTemplateExecute(instance_url, access_token, payload, api_name):
    time.sleep(2)
    return payload
    
def executeAll(instance_url:str, access_token:str, api_name:str, payload_list:list[dict], max_workers=5):
    """
    Execute promptTemplateExecute for multiple payloads in parallel.
    """
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_payload = {
            executor.submit(simulatePromptTemplateExecute, instance_url, access_token, payload, api_name): payload
            for payload in payload_list
        }

        # Process results as they complete

        with Progress() as progress:
            
            task1 = progress.add_task("[red]Running Tests...", total=len(payload_list))

            for future in concurrent.futures.as_completed(future_to_payload):
                payload = future_to_payload[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    console.print(f"[red]Failed:[/red] Payload {payload} -> {e}")

                progress.update(task1,advance=1)

    console.print("[green]:heavy_check_mark: [bold green]All tests completed![/]")

    return results

def promptRun(creds, promptTemplate):

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

    for k in inputs:

        instruction = f"({k['definition']})"
        if k['required']:
            instruction += ' (required)'
        userInput[k['apiName']] = inquirer.text(message=f"{k['masterLabel']}:",instruction=instruction).execute()


    jsonRequest = parseJsonRequestBody(promptInputs=inputs, userInputs=userInput)

    with console.status("[bold green]Running...[/]", spinner="dots"):

        jsonResponse = promptTemplateExecute(instance_url=creds['instance_url'],access_token=creds['access_token'],payload=jsonRequest,api_name=promptTemplate)

    console.print("[green]:heavy_check_mark: [bold green]Prompt template executed successfully![/]")

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
    

def promptPreview(creds,promptTemplate):

    with console.status("[bold green]Fetching metadata, please wait...[/]", spinner="dots"):

        result = subprocess.run(
            ["sf", "project", "retrieve", "start", "--metadata",f"GenAiPromptTemplate:{promptTemplate}","--json"],
            capture_output=True,
            text=True,
            check=False,
            shell=True  # Do not raise error on non-zero exit code
        )
        
    if result.returncode != 0:

        console.print("Error querying prompt template metadata")

        raise typer.Exit(code=1)
    
    console.print("[green]:heavy_check_mark: [bold green]Metadata fetched successfully![/]")

    metadata = parseXmlMetadata(promptTemplate)

    # Extract main template
    template = metadata['GenAiPromptTemplate']

    # General Info Table
    general_table = Table(box=None, show_header=False)
    general_table.add_row("Developer Name", template['developerName'])
    general_table.add_row("Master Label", template['masterLabel'])
    general_table.add_row("Type", template['type'])
    general_table.add_row("Visibility", template['visibility'])

    console.print(Panel(general_table, title="General Information", style="bold cyan", border_style="bright_blue"))

    # Template Content
    console.print(Panel(
        Align.left(template['templateVersions'][0]['content'][:300] + "...\n\n[dark_grey]Content truncated[/]"),
        title="Template Content",
        border_style="magenta",
        padding=(1, 2)
    ))

    # Generation Configs
    console.print(Panel(
        Pretty(template['templateVersions'][0]['generationTemplateConfigs']),
        title="Generation Configs",
        border_style="yellow"
    ))

    # Inputs Section as a list of objects
    inputs_table = Table(show_header=True, header_style="bold green")
    inputs_table.add_column("API Name", style="cyan", no_wrap=True)
    inputs_table.add_column("Definition", style="magenta")
    inputs_table.add_column("Master Label", style="yellow")
    inputs_table.add_column("Reference Name", style="green")
    inputs_table.add_column("Required", style="red")

    # Iterate over list of input objects
    for input_obj in template['templateVersions'][0]['inputs']:
        inputs_table.add_row(
            input_obj.get('apiName', ''),
            input_obj.get('definition', ''),
            input_obj.get('masterLabel', ''),
            input_obj.get('referenceName', ''),
            input_obj.get('required', '')
        )

    console.print(Panel(inputs_table, border_style="bright_green", title="Inputs Section"))


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

    with console.status("[bold green]Fetching Metadata, please wait...[/]", spinner="dots"):

        promptInputs = getPromptTemplateInputs(apiName=promptTemplate)

    console.print("[green]:heavy_check_mark: [bold green]Metadata fetched successfully![/]")

    home_path = os.getcwd()
    src_path = inquirer.filepath(
    message="Enter file to upload:",
    default=home_path,
    validate=lambda path:True if (os.path.isfile(path) and path.lower().endswith(".csv")) else False,
    only_files=True,
    ).execute()

    # Read CSV and build structured dicts per row

    df = pd.read_csv(src_path)

    user_inputs = []

    for i,row in df.iterrows():
        user_input = defaultdict()
        for prompt_input in promptInputs:
            user_input[prompt_input['apiName']] = row[prompt_input['apiName']]
        user_inputs.append((prompt_input,user_input))

    payload_list = []

    for prompt_input,user_input in user_inputs:

        payload = parseJsonRequestBody(promptInputs=promptInputs,userInputs=user_input)

        payload_list.append(payload)

    results = executeAll(instance_url=creds['instance_url'],access_token=creds['access_token'],api_name=promptTemplate,payload_list=payload_list)

    console.print(results)
    
    
            