#!/usr/bin/env python3
"""Extract Gutenberg content from downloaded file."""

def extract_gutenberg_content(source_filename, target_filename):
    """
    Extract content between Gutenberg markers.
    """
    try:
        with open(source_filename, 'r', encoding='utf-8') as infile, \
             open(target_filename, 'w', encoding='utf-8') as outfile:
            inside_content = False
            start_marker = '*** START OF THE PROJECT GUTENBERG EBOOK'
            end_marker = '*** END OF THE PROJECT GUTENBERG EBOOK'

            for line in infile:
                if not inside_content:
                    if start_marker in line:
                        inside_content = True
                        continue
                else:
                    if end_marker in line:
                        inside_content = False
                        break
                    outfile.write(line)

        print(f"Successfully extracted content to '{target_filename}'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    extract_gutenberg_content('100.txt.utf-8', 'Shakespeare.txt')
