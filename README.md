# AtmosGen - AI Weather Forecasting

Physics-Informed Diffusion Model for Satellite Weather Nowcasting

## Features

- **AI Weather Prediction**: Advanced diffusion model for weather forecasting
- **Satellite Data Integration**: Real-time satellite image processing
- **User Authentication**: Secure user accounts and session management
- **Forecast History**: Save and manage weather predictions
- **Real-time Dashboard**: Track prediction statistics and history
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AtmosGen.git
   cd AtmosGen
   ```

2. **Start Backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```

3. **Start Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Deploy to Production

See [docs/DEPLOY_NOW.md](docs/DEPLOY_NOW.md) for quick deployment guide.

## Architecture

- **Backend**: FastAPI with Python
- **Frontend**: React with TypeScript
- **Database**: SQLite (local) or Supabase (production)
- **AI Model**: Custom diffusion model for weather prediction
- **Authentication**: JWT-based with session management

## Database Options

- **SQLite**: Local development and testing
- **Supabase**: Production deployment with real-time features

See [docs/SUPABASE_SETUP.md](docs/SUPABASE_SETUP.md) for database configuration.

## API Endpoints

- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user
- `POST /predict` - Generate weather forecast
- `GET /forecasts` - Get user's forecast history
- `GET /satellite/sequence` - Get satellite image data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support, please open an issue on GitHub.