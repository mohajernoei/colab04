"""
Deprecated no-op placeholder.

The Colab version does not run a background server. Prediction happens directly
inside utils.predict_with_target_model(...).
"""

if __name__ == "__main__":
    print(
        "background_process.py is deprecated. "
        "No Flask/server process is needed in the Colab version."
    )
