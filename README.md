# AtmosGen - Production Ready System
### Physics-Informed Diffusion Model for Real-Time Satellite Weather Nowcasting

AtmosGen is now a production-ready full-stack application for satellite weather nowcasting using deep generative models.

---

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory:**
```bash
cd backend
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the backend server:**
```bash
python main.py
```

The backend will start on `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
```bash
cd frontend
```

2. **Install Node.js dependencies:**
```bash
npm install
```

3. **Start the development server:**
```bash
npm run dev
```

The frontend will start on `http://localhost:3000`

---

## Project Structure

```
atmosgen/
│
├── backend/                    # FastAPI backend
│   ├── main.py                # FastAPI application
│   ├── model_service.py       # Model loading and inference
│   ├── database.py            # SQLite database operations
│   ├── schemas.py             # Pydantic schemas
│   ├── utils.py               # Utility functions
│   └── requirements.txt       # Python dependencies
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── app/               # Main application components
│   │   ├── lib/               # Utilities and API client
│   │   └── styles/            # CSS and styling
│   ├── package.json           # Node.js dependencies
│   └── vite.config.ts         # Vite configuration
│
├── core_model/                 # ML model code
│   ├── models/                # Neural network architectures
│   ├── data/                  # Data loading and preprocessing
│   ├── config.py              # Model configuration
│   ├── sampling.py            # Inference sampling
│   ├── trainer.py             # Training logic
│   └── metrics.py             # Evaluation metrics
│
├── checkpoints/               # Model checkpoints
├── main.py                    # Training script
└── README.md                  # This file
```

---

## API Endpoints

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Authentication
```bash
POST /auth/register
POST /auth/login
POST /auth/logout
GET /auth/me
```

### Generate Forecast
```bash
POST /predict
```

**Request:** Multipart form data with image files

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  -F "files=@image3.png" \
  -F "forecast_name=My Forecast"
```

**Response:**
```json
{
  "success": true,
  "generated_image": "base64_encoded_image_data",
  "processing_time": 2.34,
  "message": "Forecast generated successfully",
  "forecast_id": 123
}
```

### Forecast History
```bash
GET /forecasts
GET /forecasts/{id}
```

---

## Usage

1. **Start both backend and frontend servers** (see Quick Start above)

2. **Open your browser** to `http://localhost:3000`

3. **Create an account:**
   - Click "Sign up" 
   - Fill in username, email, and password
   - You'll be automatically logged in

4. **Upload satellite images:**
   - Click "Select Images" button
   - Choose 1-10 satellite images in sequence
   - Preview will show uploaded images

5. **Generate forecast:**
   - Click "Generate Forecast" button
   - Wait for processing (loading indicator will show)
   - View the generated future frame

6. **View forecast history:**
   - Go to Dashboard to see all your saved forecasts
   - Click on any forecast to view details

---

## Model Details

- **Architecture:** UNet-based conditional diffusion model
- **Input:** Sequence of satellite images (128x128 RGB)
- **Output:** Single future frame prediction
- **Inference:** CPU/GPU compatible with automatic fallback
- **Processing:** Optimized for single-batch inference

---

## Development

### Training the Model
```bash
python main.py
```

### Backend Development
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend
npm run dev
```

---

## Features

**Full-stack web application**  
**Real-time inference API**  
**Interactive React frontend**  
**Multi-image sequence upload**  
**User authentication and accounts**  
**Forecast history and management**  
**Loading indicators and error handling**  
**Base64 image response format**  
**CORS support for development**  
**Health monitoring endpoint**  
**Production-ready structure**  

---

## Phase 2 Complete

- User authentication system
- SQLite database integration  
- Session management
- Forecast history storage
- Modern glassmorphic UI
- Mobile-responsive design
- Error handling and validation

---

## Next Phases

- **Phase 3:** Real satellite data integration & online hosting
- **Phase 4:** Advanced features & optimization
- **Phase 5:** Scaling & community features

---

## Author

**Rishabh Ranjan Singh**  
- GitHub: [Rishabh1925](https://github.com/Rishabh1925)  
- LinkedIn: [Rishabh Ranjan Singh](https://www.linkedin.com/in/rishabh-ranjan-singh)

---

## License

Research prototype - under active development.
