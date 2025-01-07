# RSSCN7 CIBR with Features Re-weighting

## Project Overview
This project implements a Content-Based Image Retrieval (CIBR) system using advanced feature re-weighting techniques on the RSSCN7 dataset. The system provides a sophisticated image search and indexing solution with enhanced feature extraction and matching.

## Key Features
- Advanced feature re-weighting methodology
- Content-based image retrieval
- RSSCN7 dataset implementation
- Full-stack web application with backend API and frontend interface

## Prerequisites
- Python 3.8+
- Node.js 14+
- Git
- pip
- npm

## System Requirements
- Operating System: Linux/macOS/Windows
- Minimum RAM: 8GB
- Disk Space: 5GB free

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/salma31nidar/-un-syst-me-d-indexation-et-de-recherche-d-images
cd -un-syst-me-d-indexation-et-de-recherche-d-images
mkdir uploadSearch
```

### 2. Backend Setup
1. Create and activate a virtual environment:
```bash
python3 -m venv venv
# For Unix/macOS
source venv/bin/activate
# For Windows
venv\Scripts\activate
```


2. Generate characteristic vectors:
```bash
python3 initialize.py
```

3. Start the Backend Server:
```bash
uvicorn main:app --reload
```

### 3. Frontend Setup
1. Navigate to frontend directory and install dependencies:
```bash
cd content-based-image-searc
npm install
npm run build
```

2. Start the Frontend Development Server:
```bash
npm start
```

## Project Structure
```
project-root/
│
├── backend/              # Backend Python files
│   ├── main.py           # Main FastAPI application
│   ├── initialize.py     # Feature vector generation script
│   └── requirements.txt  # Python dependencies
│
├── content-based-image-searc/  # Frontend directory
│   ├── src/              # React/Vue source files
│   ├── package.json      # Frontend dependencies
│   └── public/           # Public assets
│
└── uploadSearch/         # Image upload directory for each users
```

## Performance Considerations
- Initial feature vector generation might take considerable time depending on dataset size

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvements`)
3. Commit changes (`git commit -m 'Enhance feature extraction'`)
4. Push to branch (`git push origin feature/improvements`)
5. Open a Pull Request
