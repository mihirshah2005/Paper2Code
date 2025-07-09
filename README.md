# 📝 PaperCoder: Smart Research Assistant

PaperCoder is an **AI-driven research automation system** designed to assist researchers in planning, coding, and evaluating machine learning experiments. This project integrates **Large Language Models (LLMs)** and iterative feedback loops to refine code generation pipelines and enhance downstream system performance.  

This work was part of a research project at the **National University of Singapore** exploring how feedback loops and prompt engineering can optimize AI-driven research assistants.  

---

## 🚀 Features
### 🔹 1. Stage-aware Critique Analysis
Classifies system critiques into three stages:  
- 🧠 **Planning**: Checks for missing or incorrect high-level research designs (e.g., wrong algorithm selection).  
- 🧪 **Analyzing**: Detects issues in initial code generation, data processing logic, and module implementation.  
- 📊 **Evaluating**: Identifies faults in evaluation metrics, test cases, and reporting.

✅ **Why it matters:**  
This allows PaperCoder to **pinpoint failure sources** and optimize stage-specific processes for better overall code quality.  

---

### 🔁 2. Feedback Loop Integration
Introduces an **iterative refinement mechanism** in the analyzing phase:  
1. Collects critiques from the first code generation pass.  
2. Feeds them back into the LLM with tailored prompts.  
3. Generates improved code for downstream evaluation.  

✅ **Why it matters:**  
Mimics human researchers refining their code after peer review, enabling PaperCoder to "learn" from its mistakes within one generation cycle.  

---

### 🤖 3. LLM Prompt Engineering
Customizes prompts sent to the LLM for each stage:  
- In **Analyzing**, prompts request modular, maintainable code with explicit comments and alignment to paper methodologies.  
- Added **severity-level tagging** (High, Medium, Low) for critiques to prioritize critical fixes.

✅ **Why it matters:**  
Improves first-try correctness and reduces cascading errors into later stages.  

---

### 📊 4. Evaluation Pipeline with Dynamic Stage-Level Fault Analysis
Benchmarks generated code against gold-standard repositories:  
- Compares implementation details.  
- Tracks severity levels of critiques.  
- Outputs structured JSON and LaTeX reports summarizing results.  

✅ **Why it matters:**  
Provides an **objective measure** of code fidelity and highlights areas for improvement.  

---

## 🏆 Achievements
- Integrated and deployed as part of a research project on **AI-driven research automation with iterative refinement methods**.  
- Currently being drafted into a research paper for publication.  

---

## 🛠 Tech Stack
- **Python**: Core backend logic  
- **OpenAI API**: LLM-based code generation  
- **tqdm, argparse**: CLI tooling and progress tracking  
- **AWS S3**: (Optional) Artifact storage and retrieval  

---

## 📂 Project Structure
PaperCoder/
├── analyzing/
│ ├── analyzing.py
│ ├── feedback_loop.py
├── utils/
│ ├── extract_planning.py
│ ├── evaluation.py
├── outputs/
│ └── generated_code/
├── README.md


---

## 📜 License
[MIT License](LICENSE)

---

## 👤 Author
**Mihir Shah**  
📧 mihirsunilshah@gmail.com  
🌐 [LinkedIn](https://linkedin.com/in/mihirsunilshah) • [GitHub](https://github.com/mihirshah2005)
