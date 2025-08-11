# Twitter Stance Detection with Gemini API

This project provides a comprehensive framework for analyzing multi-modal (text and image) Twitter data to determine stance on various topics. It leverages the power of the Gemini API for advanced language and image understanding.

This project is designed for developers and researchers who want to perform stance detection on Twitter data using state-of-the-art language models. It provides a flexible and extensible framework that can be easily adapted to different datasets and use cases.

## Key Features

*   **Multi-modal Stance Detection:** Analyzes both tweet text and images to determine stance, see `src/harness/labeling.py` for details.
*   **Topic Filtering:** Focus the analysis on specific topics of interest by using the `exclude_topics` parameter in the `process_dataset` method in `src/dataset/data_processing.py`.
*   **Stance Normalization:** Automatically normalizes different stance labels into a consistent set of categories (Pro, Against, Neutral, Unrelated) using the `STANCE_MAPPING` in `src/config.py`.
*   **Stance Switching:** Automatically handles opposing targets (e.g., Pro-Trump vs. Against-Biden) by using the `STANDARDIZED_TOPICS` dictionary in `src/config.py` and the `standardize_annotation_target` function in `src/dataset/data_processing.py`.
*   **Caching:** Caches the results of API calls to reduce costs and speed up analysis, see `src/harness/llm_cache.py` for the implementation.
*   **Comprehensive Evaluation:** Provides detailed evaluation metrics, including accuracy, F1-score, and performance by topic, see `src/dataset/evaluation.py`.
*   **Jupyter Notebook Interface:** Provides an easy-to-use interface for running the analysis and visualizing the results in `Twitter_Stance_Analysis.ipynb`.

## Getting Started

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Key:**

    Create a `.env` file in the root directory and add your Gemini API key:

    ```
    GEMINI_API_KEY="your_gemini_api_key_here"
    ```

3.  **Run the Notebook:**

    Start the Jupyter notebook server and open `Twitter_Stance_Analysis.ipynb`.

    ```bash
    jupyter notebook
    ```
