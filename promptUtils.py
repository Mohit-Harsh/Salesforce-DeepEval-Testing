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
import typer
import json
import time
import html
import re
import concurrent.futures
import csv
from deepeval.test_case import LLMTestCase
from deepEval import deepEvalTest
import xmltodict

console = Console()

def promptTemplateMetadata(promptTemplate: str) -> dict:

    # Parse XML and handle namespaces

    dir = os.path.join(os.getcwd(),f'force-app/main/default/genAiPromptTemplates/{promptTemplate}.genAiPromptTemplate-meta.xml')

    with open(dir, "r", encoding="utf-8") as f:
        xml_content = f.read()

    metadata = xmltodict.parse(xml_content)

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
    
    templateVersions = promptTemplateMetadata(promptTemplate=apiName)['GenAiPromptTemplate']['templateVersions']

    inputs = []

    if type(templateVersions) == dict:

        inputs = templateVersions['inputs']

    elif type(templateVersions) == list:

        inputs = templateVersions[0]['inputs']

    if type(inputs) == list:

        return inputs

    return [inputs]


def parseJsonRequestBody(promptInputs, userInputs):
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


def promptTemplateExecute(instance_url:str, access_token:str, payload:list[dict], api_name:str):
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
    
def executeAll(instance_url:str, access_token:str, api_name:str, payload_list:list[dict], max_workers=5):
    """
    Execute promptTemplateExecute for multiple payloads in parallel.
    """
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_payload = {
            executor.submit(promptTemplateExecute, instance_url, access_token, payload, api_name): payload
            for payload in payload_list
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_payload):
            payload = future_to_payload[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                console.print(f"[red]Failed:[/red] Payload {payload} -> {e}")
                results.append({"payload": payload, "error": str(e)})

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

        jsonResponse = executeAll(instance_url=creds['instance_url'],access_token=creds['access_token'],payload_list=[jsonRequest],api_name=promptTemplate)[0]

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

    console.print("[green]:heavy_check_mark: [bold green]Metadata fetched successfully![/]")
        
    if result.returncode != 0:

        console.print("Error querying prompt template metadata")

        raise typer.Exit(code=1)

    metadata = promptTemplateMetadata(promptTemplate)

    # Extract metadata
    template = metadata['GenAiPromptTemplate']
    master_label = template['masterLabel']
    developer_name = template['developerName']
    inputs=[]
    if type(template['templateVersions']) == dict:
        inputs = template['templateVersions']['inputs']
    elif type(template['templateVersions']) == list:
        inputs = template['templateVersions'][0]['inputs'] 

    # Create a header panel
    console.print(Panel.fit(f"[bold blue]GenAI Prompt Template Metadata[/bold blue]", style="cyan", padding=(1, 2)))

    # Create a summary table
    summary_table = Table(title="Basic Information", box=box.SIMPLE_HEAVY)
    summary_table.add_column("Field", style="bold green", no_wrap=True)
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Master Label", master_label)
    summary_table.add_row("Developer Name", developer_name)

    # Display summary
    console.print(summary_table)

    # Create a detailed inputs table
    inputs_table = Table(title="Inputs Metadata", box=box.MINIMAL_DOUBLE_HEAD)
    inputs_table.add_column("Attribute", style="bold yellow", no_wrap=True)
    inputs_table.add_column("Value", style="white")

    for key, value in inputs.items():
        inputs_table.add_row(key, str(value))

    # Display inputs
    console.print(inputs_table)


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