import time
from rich.console import Console
import xmltodict
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.pretty import Pretty
from promptUtils import *

console = Console()

# with console.status("[bold green]Loading, please wait...[/]", spinner="dots"):
    # Simulate some background task

with Progress() as progress:

    task1 = progress.add_task("[red]Downloading...", total=1000)
    task2 = progress.add_task("[green]Processing...", total=1000)
    task3 = progress.add_task("[cyan]Cooking...", total=1000)

    while not progress.finished:
        progress.update(task1, advance=0.5)
        progress.update(task2, advance=0.3)
        progress.update(task3, advance=0.9)
        time.sleep(0.02)

console.print("[green]:heavy_check_mark: [bold green]Task completed successfully![/]")

# with Progress(
#         SpinnerColumn(),
#         TextColumn("[green][progress.description]{task.description}"),
#         console=console,
#     ) as progress:

#     progress.add_task('Loading, Please wait...',start=True)

#     time.sleep(1)

# console.print("[green]:heavy_check_mark: [bold green]Task completed successfully![/]")

