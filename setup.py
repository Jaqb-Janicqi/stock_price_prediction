import os
import subprocess


dependency_dir = 'dependencies'
if not os.path.exists(dependency_dir):
    os.makedirs(dependency_dir)
os.chdir(dependency_dir)

# dependencies
dependencies = []
dependencies.append('https://github.com/keithorange/PatternPy')

for dep in dependencies:
    subprocess.run(['git', 'clone', dep])
    os.chdir(dep.split('/')[-1])
    os.chdir('..')
