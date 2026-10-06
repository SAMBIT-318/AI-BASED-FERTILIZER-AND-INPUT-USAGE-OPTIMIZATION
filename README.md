# Smart Kishan : AI-Based Precision Agronomy & Control Center

Smart Kishan is an AI-powered agricultural mobile-responsive application integrating Google Gemini AI and 5 custom-trained Machine Learning pipelines to replace farming guesswork with data-driven agronomy.

## 🌟 Key Features
1. **Gemini AI Chatbot Integration**: Sidebar conversational AI for on-the-spot plant pathology and chemical formulation support.
2. **5 Machine Learning Models**: 
   - Crop Recommender (ExtraTreesClassifier)
   - Fertilizer Predictor (RandomForestClassifier)
   - Yield Regressor (RandomForestRegressor)
   - Irrigation & Weather Risk Predictor (RandomForestRegressor)
   - Future Market Price Predictor (LinearRegression)
3. **Real-time Live Dashboards**: Global seed variants, real-time simulated weather arrays, and price market trends.
4. **4R Nutrient Stewardship & Cost Optimization**: Linear programming algorithm enforcing budget compliance.
5. **Computer Vision Pathology**: Optical leaf analysis and genuine soil checking.

## 🚀 Setup & Installation
```bash
git clone [https://github.com/your-repo/smart-kishan.git](https://github.com/your-repo/smart-kishan.git)
cd smart-kishan

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the training pipeline to generate .pkl files from your raw CSV data
python train_pipeline.py

# Launch the Application
streamlit run app.py
