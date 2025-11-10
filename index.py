import typer
import os
import subprocess
from rich.console import Console
from app import main

console = Console()

app = typer.Typer(help="DeepEval CLI tool")

@app.command("create")
def create_project(
    project: str = typer.Argument(..., help="Subcommand (must be 'project')"),
    name: str = typer.Option(..., "--name", "-n", help="Name of the new project")
):
    """
    Create a new DeepEval project.
    """
    if project != "project":
        console.print("Error: Unknown resource. Only 'project' is supported.")
        raise typer.Exit(code=1)

    project_dir = os.path.join(os.getcwd(), name)

    if os.path.exists(project_dir):
        console.print(f"❌ Project '{name}' already exists.")
        raise typer.Exit(code=1)

    os.makedirs(project_dir)
    console.print(f"✅ Created DeepEval project: {project_dir}")


@app.command("start")
def start_project(project: str = typer.Argument(..., help="Subcommand (must be 'project')")):
    """
    Start the DeepEval project (placeholder logic).
    """
    if project != "project":
        console.print("Error: Unknown resource. Only 'project' is supported.")
        raise typer.Exit(code=1)


    # If exit code is 0, it's a valid SFDX project
    if os.path.exists(os.path.join(os.getcwd(),'sfdx-project.json')):

        console.print("🚀 Starting DeepEval project...")
        main()

    else:

        console.print("❌ Not an SFDX project.")
        raise typer.Exit(code=1)   

if __name__ == "__main__":
    app()