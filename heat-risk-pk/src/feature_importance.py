"""Legacy HGB/RF permutation importance removed in the GRU-only pipeline."""


def main():
    print(
        "Tabular permutation / RF importance is disabled in the GRU-only pipeline.\n"
        "Use input×gradient saliency from:  cd heat-risk-pk && python -m src.evaluate"
    )


if __name__ == "__main__":
    main()
