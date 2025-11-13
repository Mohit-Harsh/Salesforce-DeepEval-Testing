import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import xmltodict

console = Console()

with console.status("[bold green]Loading, please wait...[/]", spinner="dots"):
    # Simulate some background task
    
    with open('./SfdxProject/force-app/main/default/genAiPromptTemplates/Human_Rights_FAQ.genAiPromptTemplate-meta.xml','r') as f:

        text = xmltodict.parse(f.read())

        print(text)

console.print("[green]:heavy_check_mark: [bold green]Task completed successfully![/]")

# with Progress(
#         SpinnerColumn(),
#         TextColumn("[green][progress.description]{task.description}"),
#         console=console,
#     ) as progress:

#     progress.add_task('Loading, Please wait...',start=True)

#     time.sleep(1)

# console.print("[green]:heavy_check_mark: [bold green]Task completed successfully![/]")

