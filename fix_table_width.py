import re

with open("Paper.tex", "r") as f:
    paper = f.read()

# Replace \begin{table}[H] or \begin{table}[htbp] with \begin{table*}[t]
paper = re.sub(r'\\begin\{table\}\[.*?\]', r'\\begin{table*}[t]', paper)

# Replace \begin{table} (if any without brackets) with \begin{table*}[t]
paper = re.sub(r'\\begin\{table\}(?!\*)', r'\\begin{table*}[t]', paper)

# Replace \end{table} with \end{table*}
paper = re.sub(r'\\end\{table\}(?!\*)', r'\\end{table*}', paper)

# Replace \resizebox{\columnwidth} with \resizebox{\textwidth}
paper = paper.replace(r'\resizebox{\columnwidth}', r'\resizebox{\textwidth}')

with open("Paper.tex", "w") as f:
    f.write(paper)

print("Tables converted to full-page width (table* with textwidth)!")
