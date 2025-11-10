from InquirerPy import inquirer
from InquirerPy.separator import Separator
from rich.markdown import Markdown
from rich.console import Console
from rich.panel import Panel
import os
from dotenv import load_dotenv
from InquirerPy.base.control import Choice
import requests

console = Console()

def getAgentList(creds):


    soql_query = "SELECT Id, MasterLabel, DeveloperName FROM BotDefinition"  # Replace with your query

    # Construct the queryAll endpoint URL
    url = f"{creds['instance_url']}/services/data/v57.0/queryAll"

    # Set up headers with authorization
    headers = {
        "Authorization": f"Bearer {creds['access_token']}",
        "Content-Type": "application/json"
    }

    # Send GET request with query as a parameter
    response = requests.get(url, headers=headers, params={"q": soql_query})

    json_response = response.json()

    result = list(map(lambda x:Choice(name=x['MasterLabel'],value=x['Id']),json_response['records']))
    
    return result

def agentPreview(creds,agent):

    console.print('You have selected: ',agent)


def agentTest(creds,agent):

    pass