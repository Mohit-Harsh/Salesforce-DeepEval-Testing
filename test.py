import time
from rich.console import Console
import xmltodict
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.pretty import Pretty

console = Console()

with console.status("[bold green]Loading, please wait...[/]", spinner="dots"):
    # Simulate some background task
    
    with open('./SfdxProject/force-app/main/default/genAiPromptTemplates/Human_Rights_FAQ.genAiPromptTemplate-meta.xml','r') as f:

        metadata = xmltodict.parse(f.read())
        if type(metadata['GenAiPromptTemplate']['templateVersions']) == dict:

            metadata['GenAiPromptTemplate']['templateVersions'] = [metadata['GenAiPromptTemplate']['templateVersions']]

        if type(metadata['GenAiPromptTemplate']['templateVersions'][0]['inputs']) == dict:

            metadata['GenAiPromptTemplate']['templateVersions'][0]['inputs'] = [metadata['GenAiPromptTemplate']['templateVersions'][0]['inputs']]
        

        # Extract main template
        template = metadata['GenAiPromptTemplate']

        # General Info Table
        general_table = Table(box=None, show_header=False)
        general_table.add_row("Developer Name", template['developerName'])
        general_table.add_row("Master Label", template['masterLabel'])
        general_table.add_row("Type", template['type'])
        general_table.add_row("Visibility", template['visibility'])
        general_table.add_row("Active Version", template['activeVersionIdentifier'])

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

        # Model & Status
        model_table = Table(box=None, show_header=False)
        model_table.add_row("Primary Model", template['templateVersions'][0]['primaryModel'])
        model_table.add_row("Status", template['templateVersions'][0]['status'])
        model_table.add_row("Version Identifier", template['templateVersions'][0]['versionIdentifier'])

        console.print(Panel(model_table, title="Model & Status", border_style="bright_white"))

        # Template Data Providers
        console.print(Panel(
            Pretty(template['templateVersions'][0]['templateDataProviders']),
            title="Template Data Providers",
            border_style="bright_magenta"
        ))

console.print("[green]:heavy_check_mark: [bold green]Task completed successfully![/]")

# with Progress(
#         SpinnerColumn(),
#         TextColumn("[green][progress.description]{task.description}"),
#         console=console,
#     ) as progress:

#     progress.add_task('Loading, Please wait...',start=True)

#     time.sleep(1)

# console.print("[green]:heavy_check_mark: [bold green]Task completed successfully![/]")

