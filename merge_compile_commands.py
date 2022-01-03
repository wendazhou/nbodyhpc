#! /bin/env python

"""Simple script to merge compile_commands for each project.
"""

import json
import os

DIRECTORIES = [
    'rasterization',
    'kdtree'
]

def main():
    result = []

    for directory in DIRECTORIES:
        compile_commands_path = os.path.join(directory, 'build', 'compile_commands.json')
        if os.path.exists(compile_commands_path):
            result.extend(json.load(open(compile_commands_path)))

    with open('compile_commands.json', 'w') as f:
        json.dump(result, f, indent=0)

if __name__ == '__main__':
    main()
