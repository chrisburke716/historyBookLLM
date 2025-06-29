# print long text with text wrapping
import textwrap


def print_with_wrapping(text, width=80):
    # Split the text by newlines to preserve paragraph structure
    paragraphs = text.split("\n")

    # Process each paragraph separately
    for paragraph in paragraphs:
        if paragraph.strip():  # Skip empty paragraphs
            # Use textwrap to wrap each paragraph to the specified width
            wrapped_lines = textwrap.wrap(paragraph.strip(), width=width)

            # Print each wrapped line
            for line in wrapped_lines:
                print(line)
