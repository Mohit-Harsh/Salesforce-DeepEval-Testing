# deep_eval_cli.py
from InquirerPy import inquirer
from rich.console import Console
from dotenv import load_dotenv
from InquirerPy.base.control import Choice
from promptUtils import *
from agentUtils import *
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from login import *

load_dotenv()

console = Console()

creds = None

def show_logo():

    logo = r"""
██████╗ ███████╗███████╗██████╗ ███████╗██╗   ██╗ █████╗ ██╗     
██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██║   ██║██╔══██╗██║     
██║  ██║█████╗  █████╗  ██████╔╝█████╗  ██║   ██║███████║██║     
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     
██████╔╝███████╗███████╗██║     ███████╗ ╚████╔╝ ██║  ██║███████╗
╚═════╝ ╚══════╝╚══════╝╚═╝     ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
"""
    console.print(Panel(logo, title="Welcome to DeepEval", subtitle="Test Salesforce Prompt Templates, Agents, and Retrievers"))

    description = """
    [bold yellow]DeepEval[/bold yellow] is a [bold cyan]Salesforce Testing CLI Tool[/bold cyan]  
    designed to help you Preview, Test, and Evaluate:
    - Einstein [green]Prompt Templates[/green]
    - [magenta]Agents[/magenta]
    - [blue]Retrievers[/blue]

    It offers automated and interactive validation of generative AI workflows,  
    helping you fine-tune prompt performance with real-time Salesforce integration.
    """
    console.print(Panel(description,border_style="cyan", title="About DeepEval", expand=False,))
    

def selectMode():
    
    choice = inquirer.select(
        message="Choose an option:",
        choices=[
            "Prompt",
            "Agent",
            "Retriever",
            "Quit"
        ],
    ).execute()

    console.print(f"\nYou selected: {choice}\n")
    # Placeholder for functionality
    return choice



def start(creds):

    mode = selectMode()

    choices = []

    # --- Modern animated spinner for loading mode data ---
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task(description="[bold cyan]Loading...", total=None)
        
        # Slight delay to simulate UI feedback and smoothness
        time.sleep(0.3)

        if mode == 'Prompt':
            progress.update(task, description="[bold green]Fetching Prompt Templates...")
            choices = getPromptList(creds)
            choices.insert(0,Choice(name='Back',value='Back'))

        elif mode == 'Agent':
            progress.update(task, description="[bold yellow]Fetching Agent Templates...")
            choices = getAgentList(creds)
            choices.insert(0,Choice(name='Back',value='Back'))

        elif mode == 'Quit':
            progress.update(task, description="[bold red]Exiting...")
            time.sleep(0.5)
            return
        
        # Allow a small delay to complete the animation visually
        time.sleep(0.5)

    # --- Main action loop ---
    while True:

        if mode == 'Prompt':

            promptTemplate = inquirer.fuzzy(
                message="Select Prompt Template:",
                choices=choices,
                max_height="70%",
            ).execute()

            if promptTemplate == 'Back':

                break

            choice = inquirer.select(
                message="Choose an action for the Prompt Template:",
                choices=[
                    "Preview",
                    "Run",
                    "Test",
                    "Back"
                ]
            ).execute()
            
            if choice == 'Preview':

                promptPreview(creds,promptTemplate)

            elif choice == 'Test':

                promptTestSingle(creds,promptTemplate)

            elif choice == 'Run':

                promptRun(creds,promptTemplate)

            elif choice == 'Back':

                break

        elif mode == 'Agent':

            agent = inquirer.fuzzy(
                message="Select Agentforce Agent:",
                choices=choices,
                max_height="70%",
            ).execute()

            if agent == 'Back':

                break

            choice = inquirer.select(
                message="Choose an action for the Agent:",
                choices=[
                    "Back",
                    "Preview",
                    "Test"
                ]
            ).execute()

            if choice == 'Preview':

                agentPreview(creds,agent)

            elif choice == 'Test':

                agentTest(creds,agent)

            elif choice == 'Back':

                break

    start(creds)


def main():

    show_logo()

    creds = login()

    if creds['logged_in']:

        start(creds)
    
        

if __name__ == "__main__":
    main()
