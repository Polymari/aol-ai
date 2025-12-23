
lines = open('d:/VSC/aol-ai/MAIN.py', 'r').readlines()

# Target Lines (1-based): 359 to 515
# Target Indices (0-based): 358 to 514
start_idx = 358
end_idx = 514

with open('d:/VSC/aol-ai/MAIN.py', 'w') as f:
    for i, line in enumerate(lines):
        if start_idx <= i <= end_idx:
             # Add 4 spaces of indentation
             if line.strip(): 
                 f.write('    ' + line)
             else:
                 f.write(line)
        else:
            f.write(line)

print("Indentation fixed.")
