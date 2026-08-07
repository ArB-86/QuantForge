from pathlib import Path

mp = Path("quantforge/walkforward/monthly.py")
code = mp.read_text()

# Find and replace the bad debug line
old = 'print("Shapes:", feature_matrix.shape, target_vector.shape, df_for_test.shape)'
new = 'print(f"Shapes: {feature_matrix.shape} {target_vector.shape} {df_for_test.shape}")'

if old in code:
    code = code.replace(old, new)
    mp.write_text(code)
    print("Debug line fixed.")
else:
    print("Old debug line not found. Printing first occurrence of 'Shapes' to help you locate it:")
    for i, line in enumerate(code.splitlines()):
        if "Shapes" in line:
            print(f"Line {i+1}: {line}")
