from ipywidgets import HTML, HBox
import numpy as np
import os
from pathlib import Path
from html2image import Html2Image
import colorsys
from functools import lru_cache
from PIL import Image, ImageChops


CIRCLE = """
    <svg class="SVG-circle">
            <circle
              cx="50%"
              cy="50%"
              r="20%"
              stroke="black"
              stroke-width="4%"
              fill="lightgray"
            ></circle>
  </svg>
"""
TASK_NOT_ASSIGNED_STRING = "1"
TASK_ASSIGNED_STRING = "0.2; filter: grayscale(50%)"


def get_color(type, index):
    # initial color schemes
    # COLORS = ["rgb(0, 119, 187)", "rgb(238, 119, 51)", "rgb(0, 153, 136)", "rgb(238, 51, 119)", "rgb(51, 187, 238)", "rgb(167, 28, 73)", "rgb(204, 51, 17)", "rgb(246, 168, 0)"]
    COLORS = [
        "#fc5c65",
        "#fd9644",
        "#fed330",
        "#26de81",
        "#2bcbba",
        "#eb3b5a",
        "#fa8231",
        "#f7b731",
        "#20bf6b",
        "#0fb9b1",
        "#45aaf2",
        "#4b7bec",
        "#a55eea",
        "#d1d8e0",
        "#778ca3",
        "#2d98da",
        "#3867d6",
        "#8854d0",
        "#a5b1c2",
        "#4b6584",
    ]
    # https://thenode.biologists.com/data-visualization-with-flying-colors/research/
    COLORS_CB = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ]
    # https://personal.sron.nl/~pault/#sec:qualitative
    COLORS_CB = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
        "#AA4499",
        "#DDDDDD",
    ]

    if type == "colorblind":
        color_code = COLORS_CB[index]
    else:
        color_code = COLORS[index]

    return color_code


def tile_row(row):
    tile_row_string = "".join(
        ["<div class='Tile'>" + CIRCLE * bool(i) + "</div>" for i in row]
    )
    return """<div class="Tile-row">""" + tile_row_string + """</div>"""


def task_tiles(assigned, task_loads, labels, task_order):
    task_tiles_string = ""
    for i in range(len(assigned)):
        i_color = task_order[i]
        task_tiles_string += f"""<div
          style="
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
          "
        >
          <span class="task-number" style="opacity: {TASK_NOT_ASSIGNED_STRING if assigned[i]==0 else TASK_ASSIGNED_STRING}{'' if labels else '; color:white'}">{task_loads[i]}</span>
          <div class="Task-tile">
            <div
              style="width: {task_loads[i]}%; background-color: {get_color("colorblind", i_color)}; opacity: {TASK_NOT_ASSIGNED_STRING if assigned[i]==0 else TASK_ASSIGNED_STRING}"
              class="Task-tile-inner"
            ></div>
          </div>
        </div>"""
    return task_tiles_string


def worker_load(assigned_tasks, task_loads, task_order, order=None):
    """
    worker_capacities : array-like
    task_loads : array-like
    order : array-like
        order of tasks inside the workers
    """
    if order is None:
        order = (-1 * task_loads).argsort()
    worker_capacities: array - like
    task_loads: array - like
    worker_load_string = ""
    for i in order:
        i_color = task_order[i]
        if assigned_tasks[i] == 1:
            worker_load_string += f"""<div class="Agent-tile-load" style="min-height: {task_loads[i]}%; background-color: {get_color("colorblind", i_color)}"></div>"""
    return """<div class="Agent-tile-load-container">""" + worker_load_string + "</div>"


def worker_tiles(
    solution_matrix, worker_capacities, task_loads, labels, task_order, order=None
):
    """
    solution_matrix : array-like
        Passes a solution matrix directly.
    worker_capacities : array-like
    task_loads : array-like
    labels : bool
        Defines if labels (load and capacity) should be displayed for workers and tasks (default=False).
    order : array-like
        order of tasks inside the workers
    """
    worker_tiles_string = ""
    for i in range(len(worker_capacities)):

        worker_tiles_string += f"""        
          <div style="display: flex; flex-direction: column">
            <span class="agent-number" style="opacity: 1{'' if labels else ';color: white'}">{worker_capacities[i] - sum(task_loads[solution_matrix[:, i].astype(bool)])}</span>
            <div class="Agent-tile">
              <div style="height: {worker_capacities[i]}%" class="Agent-tile-inner-capacity"></div>
              {worker_load(solution_matrix[:, i], task_loads, task_order, order=order) }
            </div>
          </div>        
        """
    return "<div class='Tile-row-agents'>" + worker_tiles_string + "</div>"


def html_css_string(
    worker_capacities,
    task_loads,
    labels=False,
    annotations=None,
    sm=None,
    order=None,
    return_hw=False,
    worker_order=None,
    task_order=None,
):
    """
    worker_capacities : array-like
    task_loads : array-like
    extra_metric : array-like
        first entry for key, second entry for abbreviation
    solved : bool
        True will include the assignments (default=True).
    labels : bool
        Defines if labels (load and capacity) should be displayed for workers and tasks (default=False).
    annotations : dict
        Defines if metrics should be displayed (default=None). Dictionary with entries like "metric:value".
    sm : array-like
        Passes a solution matrix directly.
    order : array-like
        order of tasks inside the workers
    """

    if worker_order is None:
        worker_order = list(range(len(worker_capacities)))
    if task_order is None:
        task_order = list(range(len(task_loads)))

    worker_capacities = worker_capacities[worker_order]
    task_loads = task_loads[task_order]

    if sm is None:
        sm = np.zeros((len(task_loads), len(worker_capacities)))
    else:
        sm = sm[task_order][:, worker_order]

    box_size = 86 * max(sm.shape)

    html_string = f"""
    <div class="App">
    <div class="problem-container">
      <div class="problem-container-left">
        <div>
          <span style='font-size:28px;color:white'>↱</span>
          <div class="Empty-tile"></div>
        </div>
        {task_tiles(sm.sum(axis=1), task_loads, labels, task_order)}
      </div>
      <div class="problem-container-middle">
        {worker_tiles(sm, worker_capacities, task_loads, labels, task_order, order)}
        <div class="Tile-container">{"".join([tile_row(row) for row in sm])}
        </div>
      </div>
      <div class="problem-container-right"></div>
    </div>
    </div>
    """

    css_string = (
        """
    body {
      margin:0;
      font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
      -webkit-font-smoothing:antialiased;
      -moz-osx-font-smoothing:grayscale
    }
    code {
      font-family:source-code-pro,Menlo,Monaco,Consolas,Courier New,monospace
    }
    :root {
      --box-size:"""
        + str(box_size)
        + """px;
      --tile-size:60px;
      --tile-size-long:120px;
      --default-font-size:medium;
      --font-mono:"Fira Mono",monospace;
      --pure-white:#fff;
      --primary-color:#b9c6d2;
      --secondary-color:#d0dde9;
      --tertiary-color:#edf0f8;
      --accent-color:#0063a6;
      --heading-color:rgba(0,0,0,.7);
      --text-color:#444;
      --selected-color:#00bfff;
      --background-without-opacity:hsla(0,0%,100%,.7);
      --column-width:42rem;
      --column-margin-top:4rem;
      font-size:var(--default-font-size)
    }

    div {
      box-sizing:border-box;
      -moz-box-sizing:border-box;
      -webkit-box-sizing:border-box
    }
    .App {
      text-align:center
      background-color:#fff;
      min-height:500px;
      font-size:calc(10px + 2vmin);
      color:#000;
    }
    .problem-container {
      display:flex;
      overflow:auto;
      background-color:#fff;
      flex-direction:row;
      overflow:hidden;
      justify-content:flex-start;
      align-content:center
    }
    .problem-container {
      height:var(--box-size);
      max-width:var(--box-size)
    }
    .problem-container-left,
    .problem-container-middle {
      display:flex;
      flex-direction:column
    }
    .problem-container-right {
      flex-direction:column;
      width:var(--tile-size-long);
      max-width:100%;
      flex-shrink:1
    }
    .Tile-container {
      flex-direction:column;
      border:1px solid gray;
      max-height:100%
    }
    .Agent-tile,
    .Tile-container {
      position:relative;
      display:flex;
      background-color:#fff
    }
    .Agent-tile {
      border:1px solid #fff;
      font-size:16pt;
      align-items:flex-end
    }
    .Agent-tile {
      width:var(--tile-size);
      height:var(--tile-size-long)
    }
    .Agent-tile-inner-capacity {
      position:absolute;
      background-color:#fff;
      border:dashed gray;
      border-width:1px 1px 0;
      width:100%;
      display:flex;
      align-items:flex-end;
      z-index:100
    }
    .Agent-tile-load-container {
      position:absolute;
      width:100%;
      height:100%;
      display:flex;
      flex-direction:column-reverse;
      justify-content:flex-start;
      z-index:110
    }
    .Agent-tile-load {
      width:100%;
      display:flex;
      align-items:flex-end;
      opacity:.95
    }
    .Task-tile {
      background-color:#fff;
      border:1px solid #fff;
      font-size:16pt;
      display:flex;
      justify-content:flex-end;
      opacity:.95
    }
    .Task-tile {
      width:var(--tile-size-long);
      height:var(--tile-size)
    }
    .Task-tile-inner {
      height:100%
    }
    .Empty-tile {
      background-color:#fff;
      border:1px solid #fff
    }
    .Empty-tile {
      width:var(--tile-size-long);
      height:var(--tile-size-long)
    }
    .Tile {
      display:flex;
      background-color:#fff;
      border:1px solid gray;
      justify-content:flex-end;
    }
    .Tile {
      width:var(--tile-size);
      height:var(--tile-size)
    }
    .Tile-row,
    .Tile-row-agents {
      display:flex;
      flex-direction:row;
      align-items:flex-end;
    }

    .task-number {
      font-size:28px;
    }

    .agent-number {
      font-size:28px;
      text-align: center;
    }

    """
    )
    if annotations != None:
        html_string = f"""<style>{css_string}</style><div style="
        display: inline-block; 
        background: white;
        text-align: center;
        ">{html_string}<span style="font-family: monospace; font-size: 26px; color: black;">{str(annotations)}</div></div>"""
    else:
        html_string = f"<style>{css_string}</style>{html_string}"

    if return_hw:
        return html_string, box_size, box_size
    else:
        return html_string


def render_html(
    worker_capacities,
    task_loads,
    labels=False,
    annotations=None,
    sm=None,
    order=None,
    worker_order=None,
    task_order=None,
):
    """
    Displays stimulus for a certain problem.

    worker_capacities : array-like
    task_loads : array-like
    labels : bool
        Defines if labels (load and capacity) should be displayed for workers and tasks (default=False).
    annotations : dict
        Defines if metrics should be displayed (default=None). Dictionary with entries like "metric:value".
    sm : array-like
        Passes a solution matrix directly. Requires solved=True.
    order : array-like
        order of tasks inside the workers
    """
    display(
        HTML(
            html_css_string(
                worker_capacities,
                task_loads,
                labels,
                annotations,
                sm=sm,
                order=order,
                worker_order=worker_order,
                task_order=task_order,
            )
        )
    )
    return None


def render_pair_html(
    worker_capacities_1,
    task_loads_1,
    worker_capacities_2,
    task_loads_2,
    labels=False,
    annotations_1=None,
    annotations_2=None,
    sm_1=None,
    sm_2=None,
    worker_order_1=None,
    task_order_1=None,
    worker_order_2=None,
    task_order_2=None,
    order_1=None,
    order_2=None,
):
    """
    Displays stimulus for a certain problem.

    worker_capacities_1, worker_capacities_2 : array-like
    task_loads_1, task_loads_2 : array-like
    labels : bool
        Defines if labels (load and capacity) should be displayed for workers and tasks (default=False).
    annotations_1/2 : dict
        Defines if metrics should be displayed (default=None). Dictionary with entries like "metric:value".
    sm_1, sm_2 : array-like
        Passes a solution matrix directly.
    order_1, order_2 : array-like
        defines the order of the tasks inside the worker
    """

    left_box = HTML(
        html_css_string(
            worker_capacities_1,
            task_loads_1,
            labels,
            annotations_1,
            sm=sm_1,
            order=order_1,
            worker_order=worker_order_1,
            task_order=task_order_1,
        )
    )
    right_box = HTML(
        html_css_string(
            worker_capacities_2,
            task_loads_2,
            labels,
            annotations_2,
            sm=sm_2,
            order=order_2,
            worker_order=worker_order_2,
            task_order=task_order_2,
        )
    )

    display(HBox([left_box, right_box]))

    return None


def create_stimulus_old(
    worker_capacities,
    task_loads,
    index,
    directory,
    type="both",
    labels=False,
    annotations=None,
    sm=None,
    order=None,
    worker_order=None,
    task_order=None,
    prefix_type="size",
    suffix_type="state",
):
    """
    Creates stimulus for a certain problem.

    worker_capacities : array-like
    task_loads : array-like
    index : int
    directory : str
    type : str
        Can be 'both' (default), 'html' or 'png'
    labels : bool
        Defines if labels (load and capacity) should be displayed for workers and tasks (default=False).
    annotations : dict
        Defines if metrics should be displayed (default=None).
    sm : array-like
        Passes a solution matrix directly. Requires solved=True.
    order : array-like
        order of tasks inside the workers
    worker_order : array-like
        order of workers along the x-axis
    task_order : array-like
        order of tasks along the y-axis
    prefix_type : str
        Can be 'size' (default) or None
    suffix_type : str
        Can be 'state' (default) or None
    """

    height = 86 * len(task_loads)
    width = max(600, 120 * len(worker_capacities))

    # quick fix to solve cut off issue
    height += 60

    if prefix_type == "size":
        prefix = f"gap_{len(task_loads)}x{len(worker_capacities)}_"
    elif prefix_type is None:
        prefix = ""
    if suffix_type == "state":
        suffix = "unsolved" if sm is None else "solved"
    elif suffix_type is None:
        suffix = ""
    a = "_annotated" if annotations != None else ""
    html_dir = Path(directory, f"html{a}")
    png_dir = Path(directory, f"png{a}")
    size = (width, height + 30) if annotations != None else (width, height)
    text, width, height = html_css_string(
        worker_capacities,
        task_loads,
        labels=labels,
        annotations=annotations,
        sm=sm,
        return_hw=True,
        order=order,
        worker_order=worker_order,
        task_order=task_order,
    )

    if (type == "both") or (type == "html"):
        if not os.path.exists(html_dir):
            os.makedirs(html_dir)
        filepath = Path(html_dir, f"{prefix}{str(index).zfill(3)}{suffix}.html")
        with open(filepath, "w") as fo:
            fo.write(text)

    if (type == "both") or (type == "png"):
        hti = Html2Image(
            output_path=png_dir,
            browser="chromium",
            custom_flags=["--hide-scrollbars", "--default-background-color=ffffff"],
        )
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)
        hti.screenshot(
            html_str=text,
            save_as=f"{prefix}{str(index).zfill(3)}{suffix}.png",
            size=size,
        )

    return None



def create_stimulus(
    worker_capacities,
    task_loads,
    *,
    index=0,
    directory=".",
    out="both",  # 'html', 'png' or 'both'
    labels=False,
    annotations=None,
    sm=None,
    order=None,
    worker_order=None,
    task_order=None,
    prefix_type="size",  # 'size' or None
    suffix_type="state",  # 'state' or None
    fname_base=None,
):
    """
    Render a worker–task instance to HTML and/or PNG.

    Args:
        worker_capacities: Capacities per worker.
        task_loads: Loads per task.
        index: Number appended to the filename (default 0).
        directory: Output root directory (default ".").
        out: 'html', 'png', or 'both' (default 'both').
        labels: Show load/capacity labels (default False).
        annotations: Extra metrics to display; adds “_annotated” (default None).
        sm: Solution matrix; controls solved/unsolved suffix (default None).
        order: Task order inside workers.
        worker_order: Worker order on the x-axis.
        task_order: Task order on the y-axis.
        prefix_type: 'size' → add "<m>x<n>_" prefix, None to skip (default 'size').
        suffix_type: 'state' → add 'solved/unsolved' suffix, None to skip (default 'state').
        fname_base: Full base filename (overrides automatic name).

    Returns:
        None
    """

    h, w = 86 * len(task_loads) + 60, max(600, 120 * len(worker_capacities))
    annotated = annotations is not None
    size = (w, h + 30) if annotated else (w, h)

    prefix = (
        f"gap_{len(task_loads)}x{len(worker_capacities)}_"
        if prefix_type == "size"
        else ""
    )
    suffix = ("unsolved" if sm is None else "solved") if suffix_type == "state" else ""
    fname = fname_base or f"{prefix}{index:03d}{suffix}"
    tag = "_annotated" if annotated else ""

    html_dir = Path(directory, f"html{tag}")
    png_dir = Path(directory, f"png{tag}")

    html, _, _ = html_css_string(
        worker_capacities,
        task_loads,
        labels=labels,
        annotations=annotations,
        sm=sm,
        return_hw=True,
        order=order,
        worker_order=worker_order,
        task_order=task_order,
    )

    if out in ("both", "html"):
        html_dir.mkdir(parents=True, exist_ok=True)
        Path(html_dir, f"{fname}.html").write_text(html)

    if out in ("both", "png"):
        png_dir.mkdir(parents=True, exist_ok=True)
        Html2Image(
            output_path=png_dir,
            browser="chromium",
            custom_flags=[
                "--hide-scrollbars",
                "--default-background-color=ffffff",
                "--force-device-scale-factor=10",
            ],
        ).screenshot(html_str=html, save_as=f"{fname}.png", size=size)


def crop_image(image_path, margin=10):
    """
    Crops an image to its content, adds a specified margin, and resizes to create a square image.

    Args:
        image_path (str): The path to the input image.
        margin (int, optional): The margin in pixels to add around the content. Default is 10.

    Returns:
        PIL.Image.Image: The cropped and resized square image with margin.

    Raises:
        FileNotFoundError: If the input image_path does not exist.
        UnidentifiedImageError: If the input file is not an image or is corrupted.

    Usage:
        cropped_image = crop_image("input_image.jpg", margin=10)
        cropped_image.show()
    """

    # Open the image
    image = Image.open(image_path)

    # Convert image to grayscale (luminance)
    grayscale_image = image.convert("L")

    # Get the bounding box of the non-background area
    bbox = ImageChops.invert(grayscale_image).getbbox()

    if bbox:
        # Crop the image to the bounding box
        cropped_image = image.crop(bbox)

        # Calculate the new size to make the image square
        max_side = max(cropped_image.width, cropped_image.height)

        # Calculate the size of the square image with margin
        new_size_with_margin = max_side + 2 * margin

        # Create a new square image with white background
        new_image = Image.new(
            "RGB", (new_size_with_margin, new_size_with_margin), color=(255, 255, 255)
        )

        # Center the cropped image on the new square image
        paste_position = (
            (new_size_with_margin - cropped_image.width) // 2,
            (new_size_with_margin - cropped_image.height) // 2,
        )

        new_image.paste(cropped_image, paste_position)

        # Save the image if an output path is provided
        new_image.save(image_path)

        return new_image

    # If the bounding box is None, return the original image
    return image
