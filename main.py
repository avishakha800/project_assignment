
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os
import json
import requests
import shutil
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn

# Set your OpenAI API key directly here (for simplicity)
# In production, you should use environment variables or a secure configuration
OPENAI_API_KEY = "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"  # Replace with your actual API key

# Initialize FastAPI app
app = FastAPI(title="CSV RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage (in a real app, you'd use a database)
csv_storage = {}

# Models
class ChatRequest(BaseModel):
    csv_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str

# Helper function to save uploaded file
def save_upload_file(upload_file: UploadFile, destination: str):
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    return destination

# Helper function to get OpenAI response
def query_openai(prompt: str) -> str:
    """Query OpenAI API with the given prompt"""
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions about CSV data."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 500
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error from OpenAI API: {response.text}"
            
    except Exception as e:
        return f"Error querying OpenAI API: {str(e)}"

# API Routes
@app.post("/upload/")
def upload_csv(file: UploadFile = File(...), description: str = Form(None)):
    """Upload a CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Save uploaded file to temp location
        temp_file_path = f"temp_{file.filename}"
        save_upload_file(file, temp_file_path)
        
        # Read with pandas
        df = pd.read_csv(temp_file_path)
        
        # Create a simple ID
        csv_id = str(len(csv_storage) + 1)
        
        # Store metadata and content
        csv_storage[csv_id] = {
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "description": description,
            "data": df.to_dict(orient="records")  # Convert to list of dictionaries
        }
        
        # Clean up temp file
        os.remove(temp_file_path)
        
        return {"message": "CSV uploaded successfully", "csv_id": csv_id}
    
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/upload-from-disk/")
def upload_from_disk(file_path: str = Form(...), description: str = Form(None)):
    """Upload a CSV from a disk location"""
    if not file_path.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found at path: {file_path}")
    
    try:
        # Read with pandas
        df = pd.read_csv(file_path)
        
        # Extract filename from path
        filename = os.path.basename(file_path)
        
        # Create a simple ID
        csv_id = str(len(csv_storage) + 1)
        
        # Store metadata and content
        csv_storage[csv_id] = {
            "filename": filename,
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "description": description,
            "source_path": file_path,
            "data": df.to_dict(orient="records")  # Convert to list of dictionaries
        }
        
        return {"message": "CSV uploaded successfully from disk", "csv_id": csv_id}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/upload-from-project/")
def upload_from_project(project_path: str = Form(...), description: str = Form(None)):
    """Upload CSVs from a project directory"""
    if not os.path.exists(project_path) or not os.path.isdir(project_path):
        raise HTTPException(status_code=404, detail=f"Directory not found: {project_path}")
    
    try:
        csv_ids = []
        for filename in os.listdir(project_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(project_path, filename)
                
                # Read with pandas
                df = pd.read_csv(file_path)
                
                # Create a simple ID
                csv_id = str(len(csv_storage) + 1)
                
                # Store metadata and content
                csv_storage[csv_id] = {
                    "filename": filename,
                    "columns": df.columns.tolist(),
                    "row_count": len(df),
                    "description": description,
                    "source_path": file_path,
                    "data": df.to_dict(orient="records")  # Convert to list of dictionaries
                }
                
                csv_ids.append({"filename": filename, "csv_id": csv_id})
        
        if not csv_ids:
            return {"message": "No CSV files found in the project directory"}
        
        return {"message": f"Processed {len(csv_ids)} CSV files", "csv_ids": csv_ids}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/csvs/")
def list_csvs():
    """List all uploaded CSV files"""
    result = []
    for csv_id, csv_info in csv_storage.items():
        # Don't include the full data in the listing
        csv_info_copy = {k: v for k, v in csv_info.items() if k != 'data'}
        result.append({
            "csv_id": csv_id,
            **csv_info_copy
        })
    return result

@app.get("/csvs/{csv_id}")
def get_csv(csv_id: str):
    """Get information about a specific CSV file"""
    if csv_id not in csv_storage:
        raise HTTPException(status_code=404, detail="CSV not found")
    
    # Don't include the full data in the response
    csv_info = csv_storage[csv_id]
    csv_info_copy = {k: v for k, v in csv_info.items() if k != 'data'}
    
    return {
        "csv_id": csv_id,
        **csv_info_copy
    }

@app.post("/chat/", response_model=ChatResponse)
def chat_with_csv(request: ChatRequest):
    """Chat with a CSV file"""
    # Check if CSV exists
    if request.csv_id not in csv_storage:
        raise HTTPException(status_code=404, detail="CSV not found")
    
    # Get CSV data
    csv_data = csv_storage[request.csv_id]
    
    # Create a simple context from the first 10 rows (or all if less than 10)
    context_rows = csv_data["data"][:10]
    context = json.dumps(context_rows, indent=2)
    
    # Create prompt for OpenAI
    prompt = f"""
    I have a CSV file with the following columns: {csv_data["columns"]}
    
    Here is a sample of the data:
    {context}
    
    Based on this data, please answer the following question:
    {request.query}
    
    If you cannot answer the question based on the provided data, please say so.
    """
    
    # Query OpenAI
    answer = query_openai(prompt)
    
    return ChatResponse(answer=answer)

@app.delete("/csvs/{csv_id}")
def delete_csv(csv_id: str):
    """Delete a CSV file"""
    if csv_id not in csv_storage:
        raise HTTPException(status_code=404, detail="CSV not found")
    
    del csv_storage[csv_id]
    return {"message": "CSV deleted successfully"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

# Run the application directly when the script is executed
if __name__ == "__main__":
    print("Starting CSV RAG API server...")
    print("API documentation available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)