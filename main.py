import jwt
from datetime import datetime, timedelta
from fastapi import FastAPI, Body, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import pandas as pd
import numpy as np
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from CalculeCaracteristics import caracGlobale
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
from pymongo import MongoClient
from pydantic import BaseModel
import bcrypt
from fastapi import Depends, HTTPException, status
from typing import Optional
from pathlib import Path


# Initialize the FastAPI app
app = FastAPI(debug=True)
app.mount("/3D Models", StaticFiles(directory="3D Models"), name="images")
app.mount("/images1", StaticFiles(directory="2D folder"), name="images1")
app.mount("/uploadSearch", StaticFiles(directory="uploadSearch"), name="uploadSearch")
BASE_2D_PATH = Path("2D folder")
BASE_3D_MODELS_PATH = Path("3D Models")
# Directory to save uploaded files
UPLOAD_DIR = "upload"
UPLOAD_DIR2 = "uploadSearch"
FEATURES_PATH = './features.csv'
FEATURE_PATH_WITH_REDUCE = './features_wih_reduce_mesh.csv'
ALPHA = 0.8
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR2, exist_ok=True)

# MongoDB client
client = MongoClient("mongodb://localhost:27017")
db = client['users']
users_collection = db['users']


# Pydantic models for user registration
class UserRegister(BaseModel):
    username: str
    password: str
    email: str
    phone: str


# Allow CORS for the frontend (localhost:3000)
origins = [
    "http://localhost:3000",  # Allow your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Autorisez votre frontend
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes
    allow_headers=["*"],  # Autorise tous les headers
)


@app.get("/download/{category}/{image_path}")
def download_image(category: str, image_path: str):
    # Construct the full file path including category
    file_path = os.path.join("RSSCN7", category, image_path)

    # Check if the file exists
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    # Serve the file
    return FileResponse(file_path, media_type="application/octet-stream", filename=image_path)


# User registration route
@app.post("/registeration")
async def register(user: UserRegister):
    # Check if the user already exists
    existing_user = users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    # Hash the password
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    # Save user to the database
    users_collection.insert_one({
        "username": user.username,
        "password": hashed_password,
        "email": user.email,
        "phone": user.phone
    })
    return JSONResponse(content={"message": "User registered successfully"}, status_code=200)


# Function to create JWT token
def create_jwt(user_id: str) -> str:
    expiration = datetime.utcnow() + timedelta(hours=1)  # Token expiration time
    payload = {"user_id": user_id, "exp": expiration}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Find the user by username
    user = users_collection.find_one({"username": form_data.username})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    # Check if the password matches
    if not bcrypt.checkpw(form_data.password.encode('utf-8'), user['password']):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    # Generate JWT token
    payload = {"user_id": str(user["_id"])}  # Adjust based on your user model
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return JSONResponse(content={"message": "Login successful", "token": token}, status_code=200)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# Secret key used for encoding and decoding the JWT
SECRET_KEY = "your_secret_key"  # Replace with your actual secret key
ALGORITHM = "HS256"  # Replace with your algorithm if different


# Function to decode the JWT and extract the user_id
def decode_jwt(token: str) -> Optional[str]:
    try:
        # Decode the token with the secret key and algorithm
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Extract the user_id (or any field you have in your JWT payload)
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# Dependency to extract the current user_id from the token
async def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    user_id = decode_jwt(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user_id


# Function to calculate distance between feature vectors (Euclidean or Cosine)
def calcule_distance(vect1, vect2):
    """
    Calculate the distance between two vectors with optional weighted distance.

    Args:
    vect1 (np.array): First feature vector
    vect2 (np.array): Second feature vector
    weights (np.array, optional): Weight array for features

    Returns:
    float: Weighted Euclidean distance
    """

    # If no weights, calculate standard Euclidean distance
    dist = np.linalg.norm(vect1 - vect2)

    return dist


@app.delete("/deleteImage")
async def delete_image(image_url: str, authorization: str = Header(None)):
    """
    Endpoint to delete a specific image uploaded by the user.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization token is missing")

    # Extract token from Authorization header
    token = authorization.split("Bearer ")[-1]

    # Decode the token and get the user_id
    user_id = decode_jwt(token)

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Extract image path from the URL
    path_parts = image_url.split("/")
    category = path_parts[-2]
    filename = path_parts[-1]

    user_dir = UPLOAD_DIR3 / Path(user_id) / category
    image_path = user_dir / filename

    # Check if the image file exists
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        # Delete the image
        os.remove(image_path)
        return JSONResponse(content={"message": "Image deleted successfully"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error deleting image")


def getSimilarImages(imagePath, obj_file_path, redcuce_mesh):
    """
    Get similar images from the database by comparing feature vectors.
    """
    # Read the uploaded image
    img = cv2.imread(imagePath)

    # Extract features for the uploaded image
    carac = caracGlobale(obj_file_path).flatten()

    # Load the features of all images from the CSV file
    if redcuce_mesh is False:
        images_features = pd.read_csv(FEATURES_PATH)
    else:
        images_features = pd.read_csv(FEATURE_PATH_WITH_REDUCE)

    features = images_features.iloc[:, 2:].values

    # Calculate the weighted distances
    distances = np.array([
        calcule_distance(carac, feat)  # Pass weights for this specific row
        for index, feat in enumerate(features)
    ])

    # Add the distances to the dataframe
    images_features['distance'] = distances

    # Sort images by distance (smallest distance = most similar)
    similar_images = images_features.sort_values(by='distance').head(10)
    print("Columns in CSV:", images_features.columns.tolist())
    # Ensure the full image path is returned
    base_url = "http://127.0.0.1:8000/images"
    similar_images['modelpath'] = similar_images['modelpath'].apply(
        lambda x: x.replace("\\", "/").lstrip("/")  # Remove leading slash and replace backslashes
    )

    similar_images = similar_images.reset_index()
    return similar_images[['index', 'modelpath', 'Category', 'distance']]

@app.get("/getImages_for_search")
async def get_images(authorization: str = Header(None)):
    """
    Endpoint to get the images uploaded by the current user, including images outside categories.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization token is missing")

    # Extract token from Authorization header
    token = authorization.split("Bearer ")[-1]

    # Decode the token and get the user_id
    user_id = decode_jwt(token)

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Check if user directory exists
    user_dir = Path(UPLOAD_DIR3) / Path(user_id)

    if not user_dir.exists():
        # If directory doesn't exist, return no images
        return JSONResponse(content={"images": [], "message": "Aucune image pour le moment"}, status_code=200)

    # Initialize the image data list
    image_data = []

    # Check for images directly in the user's directory (outside categories)
    image_files = [f for f in os.listdir(user_dir) if os.path.isfile(user_dir / f)]
    for filename in image_files:
        image_url = f"http://localhost:8000/uploadSearch/{user_id}/{filename}"
        image_data.append(
            {"url": image_url, "category": "uncategorized"})  # You can label them as "uncategorized" or similar

    # Get the categories subdirectories (i.e., each category has its own folder)
    categories = [category for category in os.listdir(user_dir) if os.path.isdir(user_dir / category)]

    if categories:
        for category in categories:
            category_dir = user_dir / category
            category_image_files = [f for f in os.listdir(category_dir) if os.path.isfile(category_dir / f)]
            for filename in category_image_files:
                image_url = f"http://localhost:8000/uploadSearch/{user_id}/{category}/{filename}"
                image_data.append({"url": image_url, "category": category})

    # Return the list of images
    return JSONResponse(content={"images": image_data})


@app.get("/getImages")
async def get_images(authorization: str = Header(None)):
    """
    Endpoint to get the images uploaded by the current user, with their category.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization token is missing")

    # Extract token from Authorization header
    token = authorization.split("Bearer ")[-1]

    # Decode the token and get the user_id
    user_id = decode_jwt(token)

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Check if user directory exists
    user_dir = UPLOAD_DIR3 / Path(user_id)

    if not user_dir.exists():
        # If directory doesn't exist, return no images
        return JSONResponse(content={"images": [], "message": "Aucune image pour le moment"}, status_code=200)

    # Get the categories subdirectories (i.e., each category has its own folder)
    categories = [category for category in os.listdir(user_dir) if os.path.isdir(user_dir / category)]

    if not categories:
        return JSONResponse(content={"images": [], "message": "Aucune image pour le moment"}, status_code=200)

    image_data = []

    for category in categories:
        category_dir = user_dir / category
        image_files = [f for f in os.listdir(category_dir) if os.path.isfile(category_dir / f)]

        for filename in image_files:
            image_url = f"http://localhost:8000/uploadSearch/{user_id}/{category}/{filename}"
            image_data.append({"url": image_url, "category": category})

    return JSONResponse(content={"images": image_data})


from pathlib import Path

UPLOAD_DIR3 = Path("uploadSearch")


async def save_file(file: UploadFile, user_id: str):
    user_dir = UPLOAD_DIR3 / Path(user_id)  # Correctly convert user_id to Path
    user_dir.mkdir(parents=True, exist_ok=True)
    file_location = user_dir / file.filename
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
    return file_location


@app.get("/uploaded_images")
async def get_uploaded_images(
        folder_name: str = "uploadSearch",
        category: str = None,  # Added category parameter
        authorization: str = Header(None)
):
    try:
        # Verify that the Authorization header is present
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization token is missing.")
        # Extract the user ID from the JWT token
        token = authorization.split("Bearer ")[-1]
        user_id = decode_jwt(token)  # Implement `decode_jwt` to decode the JWT token

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")

        # Construct the user's folder path
        user_folder = os.path.join(folder_name, user_id)

        if not os.path.exists(user_folder):
            return JSONResponse(content={"message": "Aucune image téléchargée pour cet utilisateur."}, status_code=404)

        # Check if category is provided and filter images
        if category:
            category_folder = os.path.join(user_folder, category)
            if not os.path.exists(category_folder):
                return JSONResponse(content={"message": f"Aucune image dans la catégorie {category}."}, status_code=404)

            files = [f for f in os.listdir(category_folder) if os.path.isfile(os.path.join(category_folder, f))]
            image_urls = [f"/{folder_name}/{user_id}/{category}/{file}" for file in
                          files]  # Generate user-specific URLs
        else:
            # If no category, fetch all images
            files = [f for f in os.listdir(user_folder) if os.path.isfile(os.path.join(user_folder, f))]
            image_urls = [f"/{folder_name}/{user_id}/{file}" for file in files]

        return {"images": image_urls}
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)


@app.post("/upload_Search")
async def upload_file_for_search(
    file: UploadFile = File(...),
    authorization: str = Header(None),
):
    """
    Endpoint to upload a file and search for a corresponding 2D image in the entire 2D folder.
    Files are saved in a user-specific category directory.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization token is missing")

    # Extract token from Authorization header
    token = authorization.split("Bearer ")[-1]

    # Decode the token and get the user_id
    user_id = decode_jwt(token)

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Extract the model name (without extension) from the uploaded file
    model_name = Path(file.filename).stem.lower()  # Convert to lowercase

    # Look for the corresponding image in the `2D folder` and its subdirectories
    image_extensions = ["png", "jpg", "jpeg"]  # Supported 2D image formats
    image_found = False
    matching_image_path = None
    category = None

    # Recursively search through all subdirectories of the 2D folder
    for ext in image_extensions:
        for potential_image_path in BASE_2D_PATH.rglob(f"*.{ext}"):  # Search for all files with the current extension
            if potential_image_path.stem.lower() == model_name:  # Case-insensitive comparison of file names
                # Extract the category from the subdirectory name
                category = potential_image_path.parent.name  # Get the parent directory name (category)
                # Define the user-specific category directory
                user_category_dir = f"uploadSearch/{user_id}/{category}"
                os.makedirs(user_category_dir, exist_ok=True)  # Ensure the directory exists

                # Copy the 2D image to the user-specific category directory
                destination_path = Path(user_category_dir) / potential_image_path.name
                with open(potential_image_path, "rb") as src, open(destination_path, "wb") as dst:
                    dst.write(src.read())

                image_found = True
                matching_image_path = destination_path
                break
        if image_found:
            break

    if not image_found:
        return {
            "message": f"File uploaded successfully, but no corresponding 2D image found for {model_name}."
        }

    return {
        "message": f"File uploaded successfully, and 2D image {potential_image_path.name} saved to {user_category_dir}. Category: {category}",
        "2d_image_path": f"/uploadSearch/{user_id}/{category}/{potential_image_path.name}",
        "category": category,
    }



class UploadRequest(BaseModel):
    file_path: str
    reduce_mesh: bool

@app.post("/upload")
async def upload_file(file_path: str = Body(..., embed=True), reduce_mesh: bool = Body(..., embed=True)):

    """
    Endpoint to process the file path, find the corresponding 3D model, and retrieve similar images.
    """
    # Normalize the path and replace backslashes with forward slashes
    normalized_path = os.path.normpath(file_path).replace("\\", "/")

    # Construct the full path for the uploaded image
    image_path = os.path.join(UPLOAD_DIR2, normalized_path.lstrip("/"))


    # Check if the image exists at the specified path
    if not os.path.exists(image_path):
        return {"error": "Image not found."}

    # Extract the category from the file path (e.g., "Pyxis" from "/uploadSearch/67804fa35785a55e21f60d67/Pyxis/london e 774.jpg")
    path_parts = normalized_path.lstrip("/").split("/")
    if len(path_parts) < 3:  # Expecting /uploadSearch/user_id/category/image_name
        return {"error": "Invalid file path format. Expected /uploadSearch/user_id/category/image_name."}

    upload_search_dir = path_parts[0]  # Should be "uploadSearch"
    user_id = path_parts[1]  # Not used in this context
    category = path_parts[1]  # Extract the category (e.g., "Pyxis")
    print("cat",category)
    image_name = path_parts[-1]  # Extract the image name (e.g., "london e 774.jpg")

    # Remove the file extension to get the base name (e.g., "london e 774")
    base_name = os.path.splitext(image_name)[0]

    # Construct the path to the category subdirectory in the 3D Models folder
    category_3d_path = BASE_3D_MODELS_PATH / category

    # Search for the corresponding .obj file in the category subdirectory
    obj_file_path = None
    if category_3d_path.exists() and category_3d_path.is_dir():
        for file in category_3d_path.iterdir():
            if file.is_file() and file.name.lower().startswith(base_name.lower()) and file.name.lower().endswith(".obj"):
                obj_file_path = file
                break

    if not obj_file_path:
        return {"error": f"No corresponding 3D model found for {base_name} in category {category}."}


    # Get similar images based on the uploaded image
    similar_images = getSimilarImages(image_path, str(obj_file_path), redcuce_mesh=reduce_mesh)

    def map_obj_to_image_path(obj_path):
        """
        Map an .obj file path to a corresponding .jpg or .png file in the /2D directory,
        ignoring case sensitivity.
        """
        # Ensure BASE_2D_PATH is defined in the code (e.g., BASE_2D_PATH = Path('/path/to/2D'))
        base_2d_dir = BASE_2D_PATH  # This should be the base path to the 2D folder
        base_3d_dir = BASE_3D_MODELS_PATH  # This should be the base path to the 3D models folder

        # Extract the category (parent directory) and the file name without extension from the obj path
        obj_path = Path(obj_path)
        category = obj_path.parent.name  # Get the category directory name (e.g., 'category')
        target_name = obj_path.stem.lower()  # Get the base file name without extension, in lower case

        # Construct the path to the corresponding 2D folder (2D/category)
        target_2d_dir = base_2d_dir / category

        # Check if the target directory exists
        if not target_2d_dir.exists() or not target_2d_dir.is_dir():
            return None

        # Iterate through files in the target directory
        for file in target_2d_dir.iterdir():
            # Check if the file is an image and matches the target name (case-insensitive comparison)
            if file.is_file() and file.suffix.lower() in [".jpg", ".png"]:
                if file.stem.lower() == target_name:  # Case-insensitive comparison of the file name
                    file = str(file).replace("2D folder", "")
                    return str(file)

        # Return None if no match is found
        return None

    # Construct URLs for similar images
    image_urls = []
    for image in similar_images['modelpath']:
        # Map the .obj path to a 2D image path
        image_path_2d = map_obj_to_image_path(image)
        # Replace backslashes with forward slashes
        image_path_with_forward_slashes = image_path_2d.replace("\\", "/")
        # Format the URL
        image_url = f"http://127.0.0.1:8000/images1/{image_path_with_forward_slashes}"
        image_urls.append(image_url)

    return {
        "image_path": image_path,  # Full image path for reference
        "3d_model_path": str(obj_file_path),  # Path to the corresponding 3D model
        "similar_images": similar_images.to_dict(orient='records'),  # Format similar images as needed
        "ImagePath": image_urls  # URLs for similar images with forward slashes
    }


@app.get("/categories")
async def get_categories():
    images_features = pd.read_csv(FEATURES_PATH)
    categories = images_features['Category'].unique().tolist()
    return categories


@app.post("/categories/{category}")
async def get_images_by_categories(category: str):
    image_features = pd.read_csv(FEATURES_PATH)

    if category == "Tout" or not category:
        images_path_by_categories = image_features["ImagePath"].tolist()
    else:
        images_path_by_categories = image_features[image_features['Category'] == category]["ImagePath"].tolist()

    return {"images": images_path_by_categories}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
