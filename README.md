# AI Pitching Analysis System

## Project Overview

The AI Pitching Analysis System is a web-based application designed to automate the biomechanical assessment of baseball pitchers. By integrating computer vision (MediaPipe) and Large Language Models (Google Gemini API), the system analyzes video footage to extract key kinematic indicators and generates expert-level, actionable feedback.

This project demonstrates a full-stack implementation, featuring a robust asynchronous backend architecture to handle computationally intensive AI inference tasks while maintaining a responsive user interface.

## Key Features

* **Automated Keyframe Detection**: Utilizes MediaPipe Pose Landmarker to identify critical pitching phases, including Peak Leg Lift, Foot Plant, Maximum External Rotation, and Ball Release.
* **Biomechanical Analysis**: Calculates joint angles and spatial relationships (e.g., knee flexion, shoulder abduction, stride length) with high precision using vector mathematics.
* **Generative AI Reporting**: Integrates Google Gemini 1.5 Flash to synthesize raw kinematic data into structured, professional coaching reports in Markdown format.
* **Asynchronous Processing Pipeline**: Implements a non-blocking architecture using Python threading, allowing for immediate file upload acknowledgment and background AI processing.
* **Cloud-Native Storage**: Leverages Google Cloud Storage (GCS) for scalable, secure, and stream-based video management.
* **Video Library Management**: Provides a comprehensive interface for searching, filtering, and managing historical analysis records.

## Technical Architecture

The system follows a decoupled client-server architecture designed for scalability and maintainability.

### Backend
* **Framework**: Python Flask
* **Database**: SQLite with SQLAlchemy ORM (for metadata and analysis results)
* **Concurrency**: Python `threading` module for background task execution.
* **API Design**: RESTful endpoints for file upload and data retrieval.

### Frontend
* **Core**: HTML5, JavaScript (ES6+)
* **Styling**: Tailwind CSS for responsive and modern UI design.
* **Interaction**: Fetch API for asynchronous data transmission and dynamic DOM updates.

### AI & Machine Learning
* **Computer Vision**: Google MediaPipe Pose (Full body landmark detection).
* **Generative AI**: Google Gemini API (Multimodal input processing).
* **Data Processing**: OpenCV and NumPy for video frame manipulation and vector calculation.

### Infrastructure
* **Storage**: Google Cloud Storage (GCS) for object storage.

## Installation and Setup

Follow these steps to set up the project locally for development or testing.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/ai-pitching-coach.git](https://github.com/YOUR_USERNAME/ai-pitching-coach.git)
cd ai-pitching-coach
2. Environment Setup
It is recommended to use a virtual environment to manage dependencies.

Bash

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Configuration
You must configure the necessary API keys and credentials before running the application.

Google Cloud Storage: Place your Service Account JSON key file in the project root (e.g., keyfile.json) and set the GOOGLE_APPLICATION_CREDENTIALS environment variable.

Gemini API: Set your API key in the environment variables or configuration file.

5. Run the Application
Bash

python app.py
The server will start at http://127.0.0.1:5000.

Project Structure
app.py: Main entry point for the Flask application, defining routes and background thread management.

analysis_model.py: Encapsulates the core AI logic, including MediaPipe inference, keyframe extraction algorithms, and Gemini API interaction.

models/: Directory containing MediaPipe task files.

templates/: HTML templates for the frontend interface.

static/: Static assets including CSS and JavaScript files.