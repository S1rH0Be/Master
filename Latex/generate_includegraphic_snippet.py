import os

root_dir = '/Users/fritz/Downloads/ZIB/Master/Treffen/BilderTreffen'

for subdir, _, files in os.walk(root_dir):
    for file in sorted(files):
        if file.lower().endswith('.png'):
            rel_path = os.path.join(subdir, file).replace('\\', '/').replace('/Users/fritz/Downloads/ZIB/Master/Treffen/BilderTreffen/', '')
            label = os.path.splitext(file)[0].replace(' ', '_')
            print(f"""\\begin{{frame}}{{}}
  \\begin{{figure}}
    \\centering
    \\includegraphics[width=0.5\\linewidth]{{{rel_path}}}
    \\caption{{}}
    \\label{{fig:{label}}}
  \\end{{figure}}
\\end{{frame}}

""")