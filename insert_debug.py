from pathlib import Path
mp = Path("quantforge/walkforward/monthly.py")
code = mp.read_text()
old = "feature_matrix = _G_FEAT\n    target_vector  = _G_TARG\n    df_for_test    = _G_DF"
new = "feature_matrix = _G_FEAT\n    target_vector  = _G_TARG\n    df_for_test    = _G_DF\n    print(\"Shapes:\", feature_matrix.shape, target_vector.shape, df_for_test.shape)"
code = code.replace(old, new)
mp.write_text(code)
print("Debug prints added.")
