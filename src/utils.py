"""
Utility functions for any project, including custom print functions with color and formatting options.
Made by me, for me, but feel free to use and modify as needed.
"""

# ANSI escape codes for text colors
text_colors = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'reset': '\033[0m'
}


def colorize(text, color='green'):
    """
    Return colored text for terminal output.
    """
    color_code = text_colors.get(color, text_colors['reset'])
    return f"{color_code}{text}{text_colors['reset']}"


def format_bold(text):
    """
    Return bold text for terminal output.

    Parameters
    -----------
    text : str
        The text to be formatted as bold.

    Returns
    --------
    str
        A formatted string with ANSI escape codes that will display the input text as bold when printed.
    """
    return f"\033[1m{text}\033[0m"


def format_underline(text):
    """
    Return underlined text for terminal output.

    Parameters
    -----------
    text : str
        The text to be formatted as underlined.

    Returns
    --------
    str
        A formatted string with ANSI escape codes that will display the input text as underlined when printed
    """
    return f"\033[4m{text}\033[0m"


def format_section_header(title, char='=', width=80, align='center'):
    """
    Format text as a section header.

    Parameters
    -----------
    title : str
        The title text to be formatted as a section header.
    char : str
        The character to use for the border (default: '=').
    width : int
        The total width of the header line (default: 80).
    align : str
        The alignment of the title text within the header (default: 'center'). Options include 'left', 'center', 'right'.

    Returns
    --------
    str
        A formatted string representing the section header, with the title centered and surrounded by lines made of the specified character.
    """
    if align == 'left':
        title = title.ljust(width)
    elif align == 'right':
        title = title.rjust(width)
    elif align == 'center':
        title = title.center(width)
    return format_bold(f'\n{char * width}\n{title}\n{char * width}')


def format_subsection_header(title, char='=', width=80, align='center'):
    """
    Format text as a subsection header.

    Parameters
    -----------
    title : str
        The title text to be formatted as a subsection header.
    char : str
        The character to use for the underline (default: '=').
    width : int
        The total width of the header line (default: 80).
    align : str
        The alignment of the title text within the header (default: 'center'). Options include 'left', 'center', 'right'.

    Returns
    --------
    str
        A formatted string representing the subsection header, with the title followed by an underline made of the specified character.
    """
    if align == 'left':
        title = title.ljust(width)
    elif align == 'right':
        title = title.rjust(width)
    elif align == 'center':
        title = title.center(width)
    return format_bold(f'{title}\n{char * width}')


def format_subsubsection_header(title, char='-', width=60):
    """
    Format text as a sub-subsection header.

    Parameters
    -----------
    title : str
        The title text to be formatted as a sub-subsection header.
    char : str
        The character to use for the underline (default: '-').
    width : int
        The total width of the header line (default: 80).

    Returns
    --------
    str
        A formatted string representing the sub-subsection header, with the title followed by an underline made of the specified character.
    """
    return f'{title}\n{char * width}'


def cprint(text, color='green', bold=True, underline=False):
    """
    Custom Print: Print text with optional color, bold, and underline formatting.

    Parameters
    -----------
    text : str
        The text to be printed.
    color : str
        The color to use for the text (default: 'green'). Options include 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    bold : bool
        Whether to print the text in bold (default: True).
    underline : bool
        Whether to underline the text (default: False).
    """
    formatted_text = colorize(text, color)
    if bold:
        formatted_text = format_bold(formatted_text)
    if underline:
        formatted_text = format_underline(formatted_text)
    print(formatted_text)


def cprintf(template, color_true='blue', color_false='red', **kwargs):
    """
    Custom Print F-string:

    Takes a template string (like "Score: {val}") and a set of variables.
    Colors the values Blue if Truthy, Red if Falsey.

    Parameters
    -----------
    template : str
        A string with placeholders for variables (e.g., "Score: {score}")
    color_true : str
        Color to use for truthy values (default: 'blue')
    color_false : str
        Color to use for falsey values (default: 'red')
    **kwargs : dict
        Key-value pairs where keys correspond to placeholders in the template and values are the variables to be colored and injected into the template.

    Example Usage
    --------------
    cprintf("Model loaded: {loaded}, Accuracy: {accuracy}, Object: {Object}", loaded=True, accuracy=0.92, Object=None)
    """
    formatted_values = {}

    for key, value in kwargs.items():
        # Wrap the string representation of the value in color codes
        formatted_values[key] = colorize(str(value), color=color_true if value else color_false)

    # Use standard .format() to inject the colored strings
    print(template.format(**formatted_values))
