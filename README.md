🧠 Student Well-being AI (Chainlit + LightGBM + SHAP + DiCE)
This project is an AI-powered chatbot that predicts student well-being risks (e.g., insomnia, stress, concentration problems) based on survey inputs. It provides explanations (SHAP) and counterfactual advice (DiCE) to help students improve their sleep and study habits. Built with Chainlit for interactive conversation.

🚀 Features
📝 Free-text input → Extracts structured features using LLM
📊 Prediction → LightGBM risk classification
🔍 Explainability → SHAP values in plain language
🔄 Counterfactuals → DiCE suggestions for actionable changes
⚡ What-if analysis → Simulate changes with custom inputs

# Docker Compose setup
└── README.md
⚡ Installation & Running
Clone the repository
git clone https://github.com/yourusername/student-wellbeing-ai.git
cd student-wellbeing-ai

Install dependencies
pip install -r requirements.txt
Prepare background data
python background.py
Run Chainlit app
chainlit run app.py

🛠 Tech Stack
Frontend/Chat: Chainlit
Model: LightGBM
Explainability: SHAP
Counterfactuals: DiCE-ML
Data Processing: Pandas, NumPy
LLM Integration: SentenceTransformers + Ollama (optional)

🧪 Example Commands in Chat
state → Show current collected features
proba → Show risk probability
shap → Explain prediction in plain English
cf → Get counterfactual advice
whatif Feature=Value → Simulate changes (e.g., whatif Sleep_Hours=8)

📖 References
SHAP Documentation
DiCE-ML
Chainlit
