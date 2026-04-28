"""Legacy entry point removed: the repo uses GRU attribution from `python -m src.evaluate` (saliency) or notebook SHAP."""


def main():
    print(
        "Tree SHAP on Random Forest is disabled in the GRU-only pipeline.\n"
        "Use:  cd heat-risk-pk && python -m src.evaluate\n"
        "Or Kernel SHAP in notebooks/deep_learning_model_selection.ipynb on the torch model."
    )


if __name__ == "__main__":
    main()
