# Rice Plant Assistant for Farmers (CPU version) 

**Rice Plant Assistant for Farmers** is an innovative application designed to support Vietnamese farmers by providing quick and reliable advice about their farmland via a multi-model chatbot. The system leverages advanced technologies to deliver tailored solutions for agriculture, addressing the unique needs of farmers in Vietnam.

## Key Features
- **Fine-tuned Vietnamese LLM:** Provides accurate and context-aware responses to farmers' textual inquiries.
- **Image-based Disease Classification:** Utilizes a classification model with ResNet152 as a feature extractor to identify wheat diseases from images.
- **User-Friendly Web Interface:** Designed to be accessible and easy to use for Vietnamese farmers.

## Abstract

Rice is a vital crop in Vietnam, supporting both agriculture and the livelihoods of millions. With the advance of technology, it is demanding that every farmer have a fast and easy way to get advice about their current state of farmland via chatbots. However, existing chatbots are typically general, not focused on a specified field like agriculture, and most of all, they are not in Vietnamese.

To address this challenge, we propose a Vietnamese web-based multi-model chatbot that integrates a fine-tuned Vietnamese LLM model for textual inputs and a classification model using ResNet152 as a feature extractor for classifying wheat disease images. By combining two models together, our approach provides an ideal chatbot for giving agricultural advice to farmers quickly and reliably.

---

## Screenshots and Videos
![Chatbot Example](https://github.com/PuxHocDL/Rice-Leaves-Assistant-For-Farmers/blob/main/Picture/demo1.jpg)

---

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- **Python 3.10 or later**
- **Pip**
- **Uvicorn**

### Steps to Set Up the Application

1. Clone the repository:
   ```bash
   git clone https://github.com/PuxHocDL/Rice-Leaves-Assistant-For-Farmers.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Rice-Leaves-Assistant-For-Farmers
   ```

3. Download the required model files:
   - Visit [Hugging Face](https://huggingface.co/PuxAI/AI002) to download the following files:
     - The fine-tuned Vietnamese LLM model (quantized in gguf format).
     - The CNN classification model.
   - Place the downloaded files in the project (Rice-Leaves-Assistant-For-Farmers).

4. Create a virtual environment:
   ```bash
   python -m venv <your_virtual_env_name>
   ```

5. Activate the virtual environment and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Start the backend server:
   ```bash
   uvicorn api:app --reload
   ```

7. Open the frontend file `index.html` in your browser to use the chatbot.


---

## How to Use
1. Download and set up the model files as instructed.
2. Upload images of diseased rice leaves to get an instant diagnosis.
3. Input your queries in Vietnamese to receive textual advice from the chatbot.
4. Leverage the integration of image and text models for comprehensive support.

---

## Contributing
We welcome contributions to improve this project! If you have suggestions for new features or find bugs, feel free to create a pull request or open an issue.

---

## Contact
For questions or feedback, please contact us at [nguyenxuanphuc010205@gmail.com](mailto:nguyenxuanphuc010205@gmail.com).
