# Movie Search App using LLM 

This project is a simple Question and Answer (QA) application built using Streamlit. The application allows users to ask questions about movies, with answers generated based on a dataset of movie titles, descriptions, and genres. It leverages OpenAI's language model for generating responses and FAISS for semantic search through the data.

## Features

- **Movie Dataset Handling**: Loads and processes a movie dataset from a CSV file.
- **Text Chunking**: Splits text into manageable chunks for processing.
- **Semantic Search**: Uses FAISS to perform semantic search on the dataset based on user queries.
- **QA Chain**: Utilizes OpenAI's language model to generate answers to user questions.
- **Interactive UI**: Built with Streamlit for a user-friendly interface.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/movie-qa-app.git
    cd movie-qa-app
    ```

2. **Create and Activate a Virtual Environment** *(Optional but recommended)*

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the Required Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Your OpenAI API Key**

   Replace `'your api key'` in the script with your actual OpenAI API key.

    ```python
    api_key = 'your api key'
    ```

## Usage

1. **Run the Streamlit App**

    ```bash
    streamlit run app.py
    ```

2. **Interact with the App**

    - Open your web browser and navigate to the local URL provided by Streamlit.
    - Enter a search query related to movies and submit it to get an answer.


## Key Components

- **app.py**: The main script that handles data loading, processing, and interaction with the user via Streamlit.
- **requirements.txt**: Lists all the Python libraries needed to run the application.

## Acknowledgements

- This application uses the [OpenAI API](https://openai.com/api/) for generating responses.
- Text embeddings and vector search are powered by [FAISS](https://github.com/facebookresearch/faiss).
- The dataset used in this application is sourced from [GitHub](https://github.com/datum-oracle/netflix-movie-titles).

## Contributing

Feel free to submit issues, fork the repository, and send pull requests to improve the application.

## License

This project is licensed under the MIT License.


