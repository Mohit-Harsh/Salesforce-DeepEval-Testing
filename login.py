# deep_eval_cli.py
from InquirerPy import inquirer
from rich.console import Console
from promptUtils import *
from agentUtils import *
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.prompt import Prompt
import subprocess, json

console = Console()


def is_logged_in():
    # --- Start a progress spinner while checking login status ---
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task(description="[bold cyan]Checking Salesforce login status...", total=None)

        # Run Salesforce CLI command to get org info
        result = subprocess.run(
            ["sf", "org", "display", "--json"],
            capture_output=True,
            text=True,
            check=False,
            shell=True  # Do not raise error on non-zero exit code
        )

    # --- After spinner completes, handle result normally ---
    if result.returncode != 0:
        console.print("[bold yellow]⚠️  Not currently logged into any Salesforce org.[/bold yellow]")
        return {"logged_in": False}

    json_response = json.loads(result.stdout)
    access_token = json_response['result'].get('accessToken')
    username = json_response["result"].get("username")
    instance_url = json_response["result"].get("instanceUrl")

    return {
        "logged_in": True,
        "username": username,
        "instance_url": instance_url,
        "access_token": access_token
    }

def login():
    status = is_logged_in()

    # --- Already logged in section ---
    if status['logged_in']:
        console.print(
            Panel.fit(
                f"✅ [bold green]You are currently logged into Salesforce[/bold green]\n"
                f"[cyan]👤 Username:[/cyan] {status['username']}\n"
                f"[cyan]🌐 Instance:[/cyan] {status['instance_url']}",
                title="[bold blue]Salesforce Login Status[/bold blue]",
                border_style="green"
            )
        )

        choice = inquirer.confirm(
            message="Would you like to continue using this org?",
            default=True
        ).execute()

        if choice:
            return status

    # --- New login flow ---
    console.print("\n[bold cyan]🔑 Please login to your Salesforce Org:[/bold cyan]\n")
    instance_url = inquirer.text(message="Instance URL:", instruction='(Optional)').execute()

    # --- Modern Rich spinner while logging in ---
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task(description="[bold yellow]Signing into Salesforce...", total=None)

        command = ["sf", "org", "login", "web", "--set-default"]

        # Add optional argument if provided
        if instance_url:
            command.extend(["--instance-url", instance_url])

        result = subprocess.run(command, check=False, shell=True, text=True, capture_output=False)

        if result.returncode != 0:

            console.print("[bold red]❌ Salesforce login failed. Please try again.[/bold red]")
            return {"logged_in": False}

        # Simulate small progress updates for smooth UI experience
        for _ in range(3):
            progress.advance(task, 1)
            import time; time.sleep(0.5)

    # --- Fetch login info after successful login ---
    with Progress(
        SpinnerColumn(spinner_name="bouncingBar"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task(description="[bold green]Fetching Salesforce Org details...", total=None)
        result = subprocess.run(
            ["sf", "org", "display", "--json"],
            capture_output=True,
            text=True,
            check=False,
            shell=True
        )

        json_response = json.loads(result.stdout)

    access_token = json_response['result'].get('access_token')
    username = json_response["result"].get("username")
    instance_url = json_response["result"].get("instanceUrl")

    console.print(
        Panel.fit(
            f"[bold green]🎉 Successfully logged into Salesforce![/bold green]\n\n"
            f"[cyan]👤 Username:[/cyan] {username}\n"
            f"[cyan]🌍 Instance URL:[/cyan] {instance_url}",
            title="[bold blue]Login Successful[/bold blue]",
            border_style="green"
        )
    )

    return {
        "logged_in": True,
        "username": username,
        "instance_url": instance_url,
        "access_token": access_token
    }
