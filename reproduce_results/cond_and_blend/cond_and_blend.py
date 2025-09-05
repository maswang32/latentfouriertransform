import argparse

# Compute Low_Highs
splits = [0.0000, 0.0078, 0.0157, 0.0313, 0.0626, 0.1251, 0.2501, 0.5001, 1.0000]
low_highs = []
low_highs += [
    [splits[0], splits[4]],
    [splits[4], splits[8]],
]

low_highs += [
    [splits[0], splits[2]],
    [splits[2], splits[4]],
    [splits[4], splits[6]],
    [splits[6], splits[8]],
]
low_highs += [
    [splits[0], splits[1]],
    [splits[1], splits[2]],
    [splits[2], splits[3]],
    [splits[3], splits[4]],
    [splits[4], splits[5]],
    [splits[5], splits[6]],
    [splits[6], splits[7]],
    [splits[7], splits[8]],
]

if __name__ == "__main__":
    all_gen_specs = []
